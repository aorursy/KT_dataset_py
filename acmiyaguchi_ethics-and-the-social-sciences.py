! pip install semanticscholar
import semanticscholar as sch

import pandas as pd

from IPython.display import display



tagged = dict(

    principles=[

        ["10.15171/ijhpm.2016.55", 3],

        ["10.1186/1471-227X-11-16", 3],

        ["10.1186/s13073-014-0106-2", 2],

        ["10.1016/j.optm.2007.04.099", 2],

        ["PMID:17132331", 3],

        ["10.1016/j.socscimed.2015.11.017", 3],

        ["10.1093/phe/phu048", 2],

        ["10.1089/hs.2015.0077", 2],

        # WHO_CDS_EPR_GIP_2007_2c.pdf

        ["10.1186/1472-6939-7-5", 3],

    ],

    practice=[

        ["10.1186/s12889-015-2617-1", 3],

        ["10.1016/j.socscimed.2015.11.017", 2],

        ["10.1186/s12910-015-0025-9", 3],

    ],

    outreach=[["10.15171/ijhpm.2016.55", 3],],

    organization=[["10.1186/s12889-015-2617-1", 3],],

    quality=[["10.1016/j.trb.2014.08.004", 3], ["10.1186/s12910-015-0025-9", 3],],

    risk=[

        ["10.1093/ofid/ofx163.1031", 1],

        ["10.1016/j.jaad.2020.03.013", 1],

        ["10.1016/j.genhosppsych.2005.04.007", 2],

        ["10.1098/rstb.2004.1483", 3],

        ["10.1186/1472-6939-7-5", 3],

        ["10.1016/j.ijnss.2016.12.002", 2],

        ["10.1177/070674370404900612", 3],

    ],

    perception=[

        ["10.1093/jtm/taaa031", 2],

        ["10.1016/j.socscimed.2007.04.022", 3],

        ["10.1098/rsif.2014.1105", 3],

        ["10.1371/currents.dis.de9415573fbf90ee2c585cd0b2314547", 2],

        ["10.1089/hs.2016.0111", 1],

        ["10.1016/j.trb.2014.08.004", 3],

        ["10.1016/j.tele.2017.12.017", 3],

        ["10.3390/ijerph13080780", 3],

        ["10.1016/j.pubrev.2019.02.006", 2],

        ["10.1177/070674370404900612", 1],

        ["10.1016/S0140-6736(20)30379-2", 2],

    ],

)



def fetch_semantic_scholar(tagged: dict) -> pd.DataFrame:

    results = []

    for tag, paper_ids in tagged.items():

        for paper_id, score in paper_ids:

            # append the semantic scholar response with other metadata

            results.append({

                **dict(tag=tag, relevanceScore=score, queryId=paper_id),

                **sch.paper(paper_id)

            })

    df = pd.DataFrame(results)

    df["numCitations"] = df["citations"].apply(len)

    df["firstAuthor"] = df["authors"].apply(lambda x: x[0]["name"].split()[-1])

    return df





def filter_results(df, tag=None):

    cols = ["tag", "firstAuthor", "year", "title", "relevanceScore", "numCitations", "influentialCitationCount", "venue", "queryId"]

    

    if tag:

        df = df.query(f"tag=='{tag}'")

    return df[cols].sort_values(["relevanceScore", "numCitations"], ascending=False)
results = fetch_semantic_scholar(tagged)

display(filter_results(results)[["firstAuthor", "year", "tag", "numCitations", "title", "queryId"]][:10])
results.sort_values(["tag", "relevanceScore", "numCitations"], ascending=False).to_csv("output.csv")
display(filter_results(results, "principles"))
display(filter_results(results, "practice"))
display(filter_results(results, "outreach"))
display(filter_results(results, "organization"))
display(filter_results(results, "quality"))
display(filter_results(results, "risk"))
display(filter_results(results, "perception"))