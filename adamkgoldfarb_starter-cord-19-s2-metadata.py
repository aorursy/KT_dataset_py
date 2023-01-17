import dask.bag as db

import itertools

import json

import numpy as np

import os

import pandas as pd

from statistics import mean, median

from tqdm.auto import tqdm
%%time



paper_metadata = db.read_text(

    "../input/s2_article_metadata-2020-04-18/*.json", files_per_partition=2000

).map(

    json.loads

)
list(paper_metadata.take(1)[0].keys())
%%time



author_metadata = db.read_text(

    "../input/s2_author_metadata-2020-04-18/*.json", files_per_partition=2000

).map(

    json.loads

)
list(author_metadata.take(1)[0].keys())
def get_semantic_author_influence(author_semantic_metadata: dict) -> dict:

    author_id = author_semantic_metadata.get("author_uid")

    author_influence = {}

    if not author_semantic_metadata.get("error"):

        author_influence[author_id] = {

            author_semantic_metadata["name"]: author_semantic_metadata[

                "influentialCitationCount"

            ]

        }

    else:

        author_influence[author_id] = "error"

    return author_influence
%%time



author_influence_list = author_metadata.map(get_semantic_author_influence).compute()

author_influence = {

    author_id: influence_values

    for author in author_influence_list

    for author_id, influence_values in author.items()

}
%%time



paper_to_author_ids = paper_metadata.map(

    lambda paper_semantic_metadata: {

        paper_semantic_metadata.get("cord_uid"): [

            author["authorId"] for author in paper_semantic_metadata.get("authors", [])

        ]

    }

).compute()
%%time



paper_to_author_names = paper_metadata.map(

    lambda paper_semantic_metadata: {

        paper_semantic_metadata.get("cord_uid"): [

            author["name"] for author in paper_semantic_metadata.get("authors", [])

        ]

    }

).compute()
paper_author_influence = []

for paper_authors in paper_to_author_ids:

    for cord_uid, author_list in paper_authors.items():

        author_influence_dict = {}

        for author_id in author_list:

            author_name_citations = author_influence.get(author_id)

            if author_name_citations != "error":

                author_influence_dict.update(author_name_citations)

            else:

                author_influence_dict.update({author_id: "error"})

        paper_author_influence.append(

            {

                "cord_uid": cord_uid,

                "author_influential_citations": author_influence_dict

            }

        )
%%time



paper_influence = paper_metadata.map(

    lambda paper: {

        "cord_uid": paper.get("cord_uid", np.nan),

        "title": paper.get("title", np.nan),

        "venue": paper.get("venue", np.nan),

        "year": paper.get("year", np.nan),

        "citation_velocity": paper.get("citationVelocity", np.nan),

        "influential_citation_count": paper.get("influentialCitationCount", np.nan),

    }

).compute()
semantic_influence_df = (

    pd.DataFrame(paper_influence)

    .set_index("cord_uid")

    .join(pd.DataFrame(paper_author_influence).set_index("cord_uid"))

)
semantic_influence_df["author_influential_citations_percentile"] = (

    semantic_influence_df["author_influential_citations"]

    .apply(lambda x: sum([i for i in x.values() if i != "error"]))

    .rank(pct=True)

)
semantic_influence_df.sort_values(by="author_influential_citations_percentile", ascending=False).head(50)
topic_frequencies = sorted(

    paper_metadata.pluck("topics", [{"topic": None}])

    .flatten()

    .pluck("topic")

    .frequencies()

    .compute(),

    key=lambda x: x[1],

    reverse=True,

)
study_design_categories = [

    "Meta analysis",

    "Randomized control trial",

    "Non-randomized trial",

    "Prospective cohort",

    "Retrospective cohort",

    "Case control",

    "Cross-sectional",

    "Case study",

    "Other",

]
study_design_keywords = list(

    itertools.chain.from_iterable(

        [design.lower().split() for design in study_design_categories]

    )

)
study_design_keywords
study_design_topics = []

for topic in topic_frequencies:

    for keyword in study_design_keywords:

        if topic not in study_design_topics and topic[0] and keyword in topic[0].lower().split():

            study_design_topics.append(topic)



study_design_topics