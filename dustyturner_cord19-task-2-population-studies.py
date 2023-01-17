!wget https://github.com/samrelins/tanda_search_qa_tool/archive/master.zip

!unzip master.zip

!wget https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/models/tanda_roberta_base_asnq.tar

%mkdir tanda_roberta_base_asnq

!tar xvf tanda_roberta_base_asnq.tar

%mv /kaggle/working/models /kaggle/working/tanda_roberta_base_asnq/

%mv /kaggle/working/tanda_search_qa_tool-master/* /kaggle/working/

%rm -rf tanda_roberta_base_asnq.tar master.zip
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
searchtool = CordSearchQATool(covid_meta = meta,

                              qa_model_dir = "tanda_roberta_base_asnq")
# search for keywords relating to resource shortages

searchtool.search(search_name="shortage", 

               containing=["shortage", "resources", "supply\W",

                           "availab", "scarce"],

               containing_threshold=2)



# display the first 3 results of the "shortage" search above

display(HTML(

    searchtool.return_html_search_results(search_name="shortage",

                                          n_results=3)

))
# collect answers to the question specified

eg_answers = searchtool.return_answers(

    search_name="shortage", 

    question="preventing or stopping shortages or supply failures",

    min_score=-2)

eg_answers[0]
# same output as above, plus html to visualise the top 5 answers

answers_1, html_answers = searchtool.return_html_answers(

    search_name="shortage", 

    question="preventing or stopping shortages or supply failures",

    min_score=-2,

    top_n = 5,

    highlight_score=-2)



display(HTML(html_answers))
searchtool.refine_search(search_name="shortage",

                         not_containing=["hearing loss", "loss of hearing"])
searchtool.search_results
# run the search again with the excluded papers

answers_1 = searchtool.return_answers(

    search_name="shortage", 

    question="preventing or stopping shortages or supply failures",

    min_score=-2)
#return a list of individual ids for each paper containing an answer / answers

cord_uids_1 = []

for cord_uid, *_ in answers_1:

    if cord_uid not in cord_uids_1:

        cord_uids_1.append(cord_uid)
summarizer_1 = CordResultSummarizer(cord_uids=cord_uids_1,

                                          meta=meta,

                                          data_dir=data_dir,

                                          tanda_dir="tanda_roberta_base_asnq")
summary_table_1 = summarizer_1.summary_table(

    challenge_question="what is the problem issue challenge", # these are defaults and can be left out as below

    solution_question="what action should be taken" 

)



display_features = ["study", "addressed_population", "strength_of_evidence", 

                    "study_type", "challenge", "solution", "journal"]

summary_table_1[display_features].head(10)
answers_2, html_answers = searchtool.return_html_answers(

    search_name="shortage", 

    question="reducing demand or use of scarce low resources",

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

    search_name="shortage", 

    question="allocating or prioritising care or scarce resources",

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



summary_table_3 = summarizer_3.summary_table()



summary_table_3[display_features].head(10)
answers_4, html_answers = searchtool.return_html_answers(

    search_name="shortage", 

    question="managing intensive care bed shortage",

    min_score=-2,

    highlight_score=-2)



display(HTML(html_answers))
cord_uids_4 = []

previous_results = cord_uids_1 + cord_uids_2 + cord_uids_3

for cord_uid, *_ in answers_4:

    if cord_uid not in cord_uids_4 and cord_uid not in previous_results:

        cord_uids_4.append(cord_uid)

        

summarizer_4 = CordResultSummarizer(cord_uids=cord_uids_4,

                                          meta=meta,

                                          data_dir=data_dir,

                                          tanda_dir="tanda_roberta_base_asnq")



summary_table_4 = summarizer_4.summary_table()



summary_table_4[display_features].head(10)
%rm -rf *



%cp ../input/hospital-infrastructure-to-prevent-outbreaks/what_are_ways_to_create_hospital_infrastructure_to_prevent_nosocomial_outbreaks.csv /kaggle/working

%cp ../input/management-of-patients-who-are-underhoused/management_of_patients_who_are_underhoused_or_otherwise_lower_social_economic_status.csv /kaggle/working

%cp ../input/methods-to-control-spread-in-communities/methods_to_control_the_spread_in_communities.csv /kaggle/working

%cp ../input/modes-of-communicating-with-target-communities/modes_of_communicating_with_target_high_risk_populations.csv /kaggle/working
summary_table = pd.concat([summary_table_1, 

                           summary_table_2,

                           summary_table_3,

                           summary_table_4])

summary_table.to_csv("what_are_recommendations_for_combating_overcoming_resource_failures.csv")