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
searchtool.search(search_name="nosocomial", 

               containing=["nosocomial"],

               containing_threshold=0)



display(HTML(searchtool.return_html_search_results(search_name="nosocomial")))
answers_1, html_answers = searchtool.return_html_answers(

    search_name="nosocomial", 

    question="arranging infrastructure to prevent outbreaks",

    min_score=-2,

    highlight_score=-2)



display(HTML(html_answers))
cord_uids_1 = []

for cord_uid, *_ in answers_1:

    if cord_uid not in cord_uids_1:

        cord_uids_1.append(cord_uid)

        

summarizer_1 = CordResultSummarizer(cord_uids=cord_uids_1,

                                          meta=meta,

                                          data_dir=data_dir,

                                          tanda_dir="tanda_roberta_base_asnq")



summary_table_1 = summarizer_1.summary_table(

    solution_question="methods to prevent outbreaks"

)



display_features = ["study", "addressed_population", "strength_of_evidence", "study_type", "challenge", "solution", "journal"]

summary_table_1[display_features].head(10)
answers_2, html_answers = searchtool.return_html_answers(

    search_name="nosocomial", 

    question="hospital policies to prevent outbreaks",

    min_score=-2,

    highlight_score=-2)



display(HTML(html_answers))
cord_uids_2 = []

for cord_uid, *_ in answers_2:

    if cord_uid not in cord_uids_2 and cord_uid not in cord_uids_1:

        cord_uids_2.append(cord_uid)

        

summarizer_2 = CordResultSummarizer(cord_uids=cord_uids_2,

                                          meta=meta,

                                          data_dir=data_dir,

                                          tanda_dir="tanda_roberta_base_asnq")



summary_table_2 = summarizer_2.summary_table()

summary_table_2[display_features].head(10)
answers_3, html_answers = searchtool.return_html_answers(

    search_name="nosocomial", 

    question="arranging departments to reduce infection risk",

    min_score=-2,

    highlight_score=-2)



display(HTML(html_answers))
cord_uids_3 = []

previous_results = cord_uids_1 + cord_uids_2

for cord_uid, *_ in answers_3:

    if cord_uid not in cord_uids_3 and cord_uid not in previous_results:

        cord_uids_3.append(cord_uid)

        

summarizer_3 = CordResultSummarizer(cord_uids=cord_uids_3,

                                          meta=meta,

                                          data_dir=data_dir,

                                          tanda_dir="tanda_roberta_base_asnq")



summary_table_3 = summarizer_3.summary_table(

    solution_question="ways to reduce risk"

)

summary_table_3[display_features].head(10)
%rm -rf *
summary_table = pd.concat([summary_table_1, summary_table_2,summary_table_3])

summary_table.to_csv("what_are_ways_to_create_hospital_infrastructure_to_prevent_nosocomial_outbreaks.csv")