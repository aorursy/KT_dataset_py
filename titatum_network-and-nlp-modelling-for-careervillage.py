import os 

from IPython.display import Image

import pandas as pd

from scipy import stats



input_dir = '../input'

eda_dir = os.path.join(input_dir,'cv-exploratory-data-analysis')

ml_gbdts_dir = os.path.join(input_dir,'cv-machine-learning-gbdts')

ml_comp_dir = os.path.join(input_dir,'cv-machine-learning-comparison')

ml_data_dir = os.path.join(input_dir,'cv-data-augmentation-text-scores')
Image(os.path.join(eda_dir, 'CumulativeCountofProfessionalsoverTime.jpg'))
Image(os.path.join(eda_dir, 'CumulativeCountofStudentsoverTime.jpg'))
Image(os.path.join(eda_dir, 'CumulativeCountofQuestionsoverTime.jpg'))
Image(os.path.join(eda_dir, '30-daywindowedActiveProfessionalsoverTime.jpg'))
Image(os.path.join(eda_dir, 'HistogramofQuestionsbyAnswerNumbers.jpg'))
Image(os.path.join(eda_dir, 'CumulativeCountofEmailsoverTime.jpg'))
Image(os.path.join(eda_dir, 'HistogramofQuestionsbyEmailNumbers.jpg'))
Image(os.path.join(eda_dir, 'questions_in_emails_per professional_in_an_active_day.jpg'))
Image(os.path.join(eda_dir, 'HistogramofUsersbyTagNumbers.jpg'))
Image(os.path.join(eda_dir, 'HistogramofQuestionsbyTagNumbers.jpg'))
examples = pd.read_parquet(os.path.join(ml_data_dir,'positive_negative_examples.parquet.gzip'))

print('The total number of instances in the ML data set is {}'.format(examples.shape[0]))

print('The total number of matched instances in the ML data set is {}'.format(examples[examples['matched']==1].shape[0]))

print('The total number of unmatched instances in the ML data set is {}'.format(examples[examples['matched']==0].shape[0]))

print('The percentage of matched instances in the ML data set is {}%'.format(

    round(100 * examples[examples['matched']==1].shape[0] / examples[examples['matched']==0].shape[0], 2)))
unmatched_data = examples[examples['matched']==0]

matched_data = examples[examples['matched']==1]



print('The p-value of the test for the difference in \'days from last activities\' means of matched and unmatched recommendations is {}.'.format(

    stats.ttest_ind(matched_data['days_from_last_activities'], unmatched_data['days_from_last_activities'], equal_var = False)[1]))



print('The p-value of the test for the difference in \'numbers of activities within 30 days\' means of matched and unmatched recommendations is {}.'.format(

    stats.ttest_ind(matched_data['professional_activities_sum_30'], unmatched_data['professional_activities_sum_30'], equal_var = False)[1]))
print('The p-value of the test for the difference in \'questioner_answerer_shared_tags\' means of matched and unmatched recommendations is {}.'.format(

    stats.ttest_ind(matched_data['questioner_answerer_shared_tags'], unmatched_data['questioner_answerer_shared_tags'], equal_var = False)[1]))
print('The p-value of the test for the difference in \'questioner_answerer_shared_tags\' means of matched and unmatched recommendations is {}.'.format(

    stats.ttest_ind(matched_data['questioners_answerers_paths'], unmatched_data['questioners_answerers_paths'], equal_var = False)[1]))
print('The p-value of the test for the difference in \'questioner_answerer_shared_tags\' means of matched and unmatched recommendations is {}.'.format(

    stats.ttest_ind(matched_data['LSI_Score'], unmatched_data['LSI_Score'], equal_var = False)[1]))
Image(os.path.join(ml_gbdts_dir, 'feature_importance.jpg'))
Image(os.path.join(ml_gbdts_dir, 'days_from_joined_dates.jpg'))
Image(os.path.join(ml_gbdts_dir, 'days_from_last_activities.jpg'))
Image(os.path.join(ml_gbdts_dir, 'questioner_answerer_shared_tags.jpg'))
Image(os.path.join(ml_gbdts_dir, 'questioners_answerers_paths.jpg'))
Image(os.path.join(ml_gbdts_dir, 'LSI_Score.jpg'))
Image(os.path.join(ml_comp_dir,'min_rank_recommendation_recall.jpg'))
Image(os.path.join(ml_comp_dir,'max_rank_recommendation_recall.jpg'))
Image(os.path.join(ml_comp_dir,'before_ml_questions_in_each_email.jpg'))
Image(os.path.join(ml_comp_dir,'after_ml_questions_in_each_email.jpg'))
import datetime as dt

test_examples = examples[examples['questions_date_added']>=dt.datetime(2018,7,1)]

print('The number of answers in the test set that correspond to an item in the matches table is {}.'.format(

    test_examples[((test_examples['matched']==1) & (test_examples['questions_date_added']!=test_examples['emails_date_sent']))].shape[0]))