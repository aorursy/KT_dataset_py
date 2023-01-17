!pip install ujson
import logging

import gc

import glob

import multiprocessing as mp

import os

from typing import Tuple



import numpy as np

import pandas as pd

from tqdm.notebook import tqdm



try:

    import ujson as json

except ImportError:

    import json





def remove_span_text(text: str, spans: list, validate: bool = False) -> str:

    """Remove 'cite/ref/eqn/..._span' from 'text'

    

    Parameters

    ----------

    text : str

    spans: list of dict

        list of `span` JSONs

    validate: bool, optional

        Check if span extracted are the same as specified; defaults to `False`

    

    Returns

    -------

    str

        text without spans

    """

    if spans:

        # TODO: vectorize...?

        chars = np.array(list(text))

        span_df = pd.DataFrame(spans)

        if validate:

            assert (

                span_df["text"]

                == span_df.apply(

                    lambda row: "".join(chars[row["start"] : row["end"]]), axis=1

                )

            ).all(), "Extracted text from `spans` is not the same!"



        mask = np.full_like(chars, True, dtype=bool)

        for _, row in span_df.iterrows():

            mask[row["start"] : row["end"]] = False



        return "".join(chars[mask])

    else:

        return text

    



def parse_section(section: str, text: str, remove_spans: bool = True, **kwargs) -> Tuple[str, str]:

    """

    Parse 'abstract' and 'body_text' sections 

    

    Parameters

    ----------

    **section data

    remove_spans: bool, optional

        Remove cite/ref/eqn/... span strings; defaults to `True`



    Returns

    -------

    tuple[str, str]

        (section, cleaned text)

    """

    spans = []

    for key, val in kwargs.items():

        if key.endswith("_spans"):

            spans.extend(val)

        else:

            logging.warning("unexpected field: `%s`", key)



    clean_str = remove_span_text(text, spans) if remove_spans else text

    return (section, clean_str)





def parse_paper_json(json_data, sup_secs: list = None) -> pd.DataFrame:

    """Parse Paper in JSON format

    

    Parameters

    ----------

    json_data:

        if not a `dict` will set as `json.load(open(json_data))`

    sup_secs: list[str], optional

        supersections to parse; defaults to `["abstract", "body_text", "back_matter"]`



    Returns

    -------

    pd.DataFrame

    """

    if not isinstance(json_data, dict):

        json_data = json.load(open(json_data))

    

    sup_secs = ["abstract", "body_text", "back_matter"]

    title_info = {

        "supsec_order": np.nan,

        "supersection": "title",

        "section_order": np.nan,

        "section": "title",

        "text": json_data["metadata"]["title"],

    }

    dfs = [pd.DataFrame([title_info])]

    for ss_idx, sup_sec in enumerate(sup_secs):

        df = pd.DataFrame(

            [parse_section(**section) for section in json_data[sup_sec]],

            columns=["section", "text"],

        )

        df.insert(0, "section_order", df.index)

        df.insert(0, "supersection", sup_sec)

        df.insert(0, "supsec_order", ss_idx)

        dfs.append(df)



    res_df = pd.concat(dfs, axis=0).reset_index(drop=True)

    res_df.insert(0, "paper_id", json_data["paper_id"])



    return res_df
# source directory; most likely = "/kaggle/input"

SOURCE_DIR = "/kaggle/input"



# source name : source path mapping

SOURCES = {

    "biorxiv": "/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/",

    "pmc": "/CORD-19-research-challenge/custom_license/custom_license/",

    "comm_use": "/CORD-19-research-challenge/comm_use_subset/comm_use_subset/",

    "noncomm_use": "/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/"

}

# Process JSON data and Export as csv

for source, path in tqdm(SOURCES.items(), desc="sources"):

    with mp.Pool() as pool:

        #print(glob.glob(os.path.join(path, "*.json")))

        files = []

        dirt = SOURCE_DIR+path

        print(dirt)

        for dirname ,_, filenames in os.walk('/kaggle/input'+path):

            for filename in filenames:

                files.append(os.path.join(dirname, filename))

                #print(os.path.join(dirname, filename))

        

        #print(files)

        src_dfs = list(

            tqdm(

                pool.imap(parse_paper_json, files),

                total=len(files),

                desc="{} files".format(source),

            )

        )



    src_df = pd.concat(src_dfs).reset_index(drop=True)

    src_df["source"] = source

    

    src_df.to_csv("{}-parsed.csv.gz".format(source), index=False, compression='gzip')
