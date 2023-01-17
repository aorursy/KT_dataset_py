!wget https://github.com/samrelins/tanda_search_qa_tool/archive/master.zip

!unzip master.zip

!wget https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_roberta_base_asnq.tar

!tar xvf tanda_roberta_base_asnq.tar

%mkdir tanda_roberta_base_asnq

%mv /kaggle/working/models/tanda_roberta_base_asnq/ckpt/* /kaggle/working/tanda_roberta_base_asnq/

%mv /kaggle/working/tanda_search_qa_tool-master/* /kaggle/working/

%rm -rf tanda_roberta_base_asnq.tar master.zip models/
import numpy as np

import pandas as pd 

from cord_search_qa_tool import CordSearchQATool

from cord_result_summarizer import CordResultSummarizer

from summarizer_helpers import *

from prep_metadata import add_missing_abstracts

from IPython.core.display import display, HTML



pd.set_option('display.max_colwidth', None)



data_dir = "/kaggle/input/CORD-19-research-challenge/"



meta = add_missing_abstracts(data_dir)



searchtool = CordSearchQATool(meta, "tanda_roberta_base_asnq")
searchtool.clear_searches()
searchtool.search(search_name="control_spread", 

               containing=["communit(y|ies)"],

               containing_threshold=0)



searchtool.refine_search(search_name="control_spread", 

               containing=["spread", "infection", "transmission"],

               containing_threshold=5)



display(HTML(searchtool.return_html_search_results(search_name="control_spread")))
control_answers, html_answers = searchtool.return_html_answers(

    search_name="control_spread", 

    question="controlling the spread of the virus in communities",

    min_score=-1,

    highlight_score=-2)



display(HTML(html_answers))
control_cord_uids = []

for cord_uid, *_ in control_answers:

    if cord_uid not in control_cord_uids:

        control_cord_uids.append(cord_uid)

        

control_summarizer = CordResultSummarizer(cord_uids=control_cord_uids,

                                          meta=meta,

                                          data_dir=data_dir,

                                          tanda_dir="tanda_roberta_base_asnq")



control_summary_table = control_summarizer.summary_table(

    solution_question="what reduces or controls virus spread in communities"

)



display_features = ["study", "addressed_population", "strength_of_evidence", "study_type", "challenge", "solution", "journal"]

control_summary_table[display_features].head(10)
compliance_keywords = ["compliance", "enforce", "comply", "adhere" 

                       "(rules|regulations|guidelines|prevention).{100}follow",

                       "follow.{100}(rules|regulations|guidelines|prevention)"]



searchtool.search(search_name="compliance", 

                  containing=compliance_keywords,

                  containing_threshold=1)



searchtool.refine_search(search_name="compliance", 

                         not_containing=["lung.{,100}compliance", 

                                         "compliance.{,100}lung"])





display(HTML(searchtool.return_html_search_results(search_name="compliance")))
comply_answers, html_answers = searchtool.return_html_answers(

    search_name="compliance", 

    question="what prevents compliance or rule following",

    min_score=-1,

    highlight_score=-2)



display(HTML(html_answers))
comply_cord_uids = []

for cord_uid, *_ in comply_answers:

    if cord_uid not in comply_cord_uids:

        comply_cord_uids.append(cord_uid)

        

comply_summarizer = CordResultSummarizer(cord_uids=comply_cord_uids,

                                         meta=meta,

                                         data_dir=data_dir,

                                         tanda_dir="tanda_roberta_base_asnq")



comply_summary_table = comply_summarizer.summary_table(

    solution_question="what improves or increases rule following or compliance"

)

comply_summary_table[display_features].head(10)
%rm -rf *
summary_table = pd.concat([control_summary_table, comply_summary_table])

summary_table.to_csv("methods_to_control_the_spread_in_communities.csv")