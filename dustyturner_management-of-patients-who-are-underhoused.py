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



pd.set_option('display.max_colwidth', None)



data_dir = "/kaggle/input/CORD-19-research-challenge/"



meta = add_missing_abstracts(data_dir)



searchtool = CordSearchQATool(meta, "tanda_roberta_base_asnq")
searchtool.clear_searches()
underhoused_queries = ["homeless", "assylyum", "drug addict", 

                       "(substance|drug|alcholol) abuse", "prison", "inmate",

                       "incarcerate", "marginali[sz]ed" "refugee", "assylum", 

                       "displaced", "poverty"]



searchtool.search(search_name="underhoused", 

                  containing=underhoused_queries,

                  containing_threshold=1)



display(HTML(searchtool.return_html_search_results(search_name="underhoused")))
answers, html_answers = searchtool.return_html_answers(search_name="underhoused", 

                                                       question="what are the best approaches",

                                                       min_score=-10,

                                                       top_n=10)

display(HTML(html_answers))
cord_uids = []

for cord_uid, *_ in answers:

    if cord_uid not in cord_uids:

        cord_uids.append(cord_uid)

        

summarizer = CordResultSummarizer(cord_uids=cord_uids,

                                  meta=meta,

                                  data_dir=data_dir,

                                  tanda_dir="tanda_roberta_base_asnq")



summary_table = summarizer.summary_table()

display_features = ["study", "addressed_population", "study_type", "challenge", "solution", "journal"]

summary_table[display_features].head(10)
%rm -rf *
summary_table.to_csv("management_of_patients_who_are_underhoused_or_otherwise_lower_social_economic_status.csv")