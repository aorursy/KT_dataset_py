import numpy as np 

import pandas as pd 

import math

import itertools

import datetime
def datasets_creating():

    pro = pd.read_csv('../input/professionals.csv')

    qs = pd.read_csv("../input/questions.csv")

    ans = pd.read_csv('../input/answers.csv')

    tags = pd.read_csv('../input/tags.csv')

    qs_tags = pd.read_csv('../input/tag_questions.csv')

    user_tag = pd.read_csv('../input/tag_users.csv')

    email = pd.read_csv('../input/emails.csv')

    match = pd.read_csv('../input/matches.csv')

    

    ans_proff = pd.merge(qs, ans, left_on='questions_id', right_on='answers_question_id')

    ans_proff = ans_proff.filter(['answers_author_id','answers_date_added']) 

    

    email_matches = pd.merge(email, match, left_on='emails_id', right_on='matches_email_id')

    email_proff = pd.merge(email_matches, qs, left_on='matches_question_id', right_on='questions_id')

    email_proff = pd.merge(email_proff, ans, left_on='questions_id', right_on='answers_question_id') 

    email_proff = email_proff.drop(['answers_body', 'questions_body', 'questions_title', 'emails_frequency_level'],axis=1)

    

    qs_tags.sort_values('tag_questions_question_id')

    qs_tagnames = pd.merge(qs_tags, tags, left_on='tag_questions_tag_id',right_on='tags_tag_id')

    qs_tagnames = qs_tagnames.drop(['tags_tag_id','tag_questions_tag_id'], axis=1)

    user_tag_exp = pd.merge(tags,user_tag, left_on='tags_tag_id', right_on='tag_users_tag_id')

    user_tag_exp = user_tag_exp.drop(['tags_tag_id','tag_users_tag_id'], axis=1)

    user_tag_exp.sort_values('tag_users_user_id')

    tag_pivot = user_tag_exp.pivot_table(values='tags_tag_name', index='tag_users_user_id', aggfunc=lambda x: " ".join(x))

    tag_pivot['tag_users_user_id'] = tag_pivot.index

    tag_pivot=tag_pivot.reset_index(drop=True)

    qs_tag_pivot = qs_tagnames.pivot_table(index='tag_questions_question_id', values='tags_tag_name', aggfunc=lambda x: " ".join(x))

    qs_tag_pivot['tag_questions_question_id']=qs_tag_pivot.index

    qs_tag_pivot = qs_tag_pivot.reset_index(drop=True)

    qs_with_tags = pd.merge(qs, qs_tag_pivot, left_on='questions_id', right_on='tag_questions_question_id')

    qs_with_tags = qs_with_tags.merge(right=ans, how='inner', left_on='questions_id', right_on='answers_question_id')

    tags = qs_with_tags.merge(right=tag_pivot, left_on='answers_author_id', right_on='tag_users_user_id')

    tags = tags.filter(['tags_tag_name_x', 'answers_author_id', 'tags_tag_name_y', 'tag_users_user_id'])

    

    return ans_proff, email_proff, tags

    

ans_proff, email_proff, df = datasets_creating()
class Professionals:

    """Selects the most suitable professionals for a particular question.



    Args:

        ans_proff (pandas.DataFrame): Questions with professional answers.

        email_proff (pandas.DataFrame): Email matches with professional, questions and professional answers.

        num_of_prof (int): The number of professionals who will receive an email.

        

    Returns:

        list: Sorted list of the most suitable professionals.

    

    """

    def __init__(self, ans_proff, email_proff, num_of_prof=10):

        

        self.num_of_prof = num_of_prof

        self.ans_proff = ans_proff

        self.email_proff = email_proff 

        self.proffessional_tags = None

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

          

        

    def fit_for_tags(self, df, ind=1):

        """Create list with relevance values of the tags, for each professional.



        Args:

            df (pandas.DataFrame): The number of professionals who will receive an email.



        """

        if ind == 1: #Fitting data for the first time

            proffessional_tags = {}

        else: #Fitting data not for the first time

            proffessional_tags = self.proffessional_tags

        



        for tags in np.array(df):

            for tag in tags[2].split():

                if tags[1] in proffessional_tags:

                    proffessional_tags[tags[1]][tag] = 1.0

                else:

                    proffessional_tags[tags[1]] = {tag: 1.0}

                    

        for proff in proffessional_tags: 

            #Professional answers on the questions.

            proff_ans = df['tags_tag_name_x'][(df['answers_author_id'] == proff)]



            for tags in proff_ans:

                for question_tag in tags.split():

                    #Increasing values of tags, that were in the questions 

                    #that were answered recently by professionals

                    if question_tag in proffessional_tags: 

                        if proffessional_tags[proff][question_tag] != 1.0: 

                            proffessional_tags[proff][question_tag] = proffessional_tags[proff][question_tag] + 0.125

                    else:

                        proffessional_tags[proff][question_tag] = 1.0



                for proff_tag in proffessional_tags[proff]:

                    if proff_tag not in tags.split():

                        if ind == 0:

                            

                            print(proff_tag)

                            print(proffessional_tags[proff][proff_tag])

                        proffessional_tags[proff][proff_tag] = proffessional_tags[proff][proff_tag] - 0.125

                        if ind == 0:

                            print(proffessional_tags[proff][proff_tag] - 0.125, proffessional_tags[proff][proff_tag])

        

        #Delete unnecessary data

        for proff in proffessional_tags.copy():

            for ex_tag in proffessional_tags[proff].copy():

                if proffessional_tags[proff][ex_tag] <= 0:

                    proffessional_tags[proff].pop(ex_tag)

                    

        self.proffessional_tags = proffessional_tags   

        

    def predict_by_tags(self, tags):

        """First Tag Module.



        Args:

            tags (str): Tags of the question.



        Returns:

            list: Sorted list of the most suitable professionals; based on their tags relevance values.



        """

        proffessional_tags = self.proffessional_tags

        candidates = []

        tags = tags.split()

        num = 0



        while len(candidates) < self.num_of_prof: 

            max_candidates = []

            

            for i in itertools.combinations(tags, len(tags)-num):

                if len(tags) == num: 

                    return max_candidates

                

                for proff in proffessional_tags:

                    ind_ = 0

                    for tag in i:

                        if str(tag) not in list(proffessional_tags[proff]):

                            ind_ = 1

                            

                    if ind_ == 0:

                        normilaze_data = 0

                        for tag in proffessional_tags[proff]:

                            tag_value = proffessional_tags[proff][tag]

                            normilaze_data = normilaze_data + tag_value



                        normilaze_data = 1 / (1 + np.exp(-normilaze_data))

                        candidates.append([normilaze_data, proff])

                        

                candidates = sorted(candidates, key=lambda x: x[0]) 

                candidates.reverse()     

                

                if len(max_candidates) < len(candidates):

                    max_candidates = candidates



            num = num + 1



        return candidates

        

        

    def predict_by_activity(self, proff, max_last_activity=500):

        """Second Activity Module.



        Args:

            proff (str): List of professionals preprocessed by First Activity module.

            max_last_activity (int): Last date of professional activity must be leass than max_last_activity ago.



        Returns:

            list: Sorted list of the most suitable professionals.



        """

        

        ans_num = {}

        ans_email_num = {}

        ans_email_time = {}

        last_day_activity = {}



        for i in proff:

            email_ind = [0.5, 1]



            proff_all_ans = self.ans_proff[:][(self.ans_proff["answers_author_id"] == i)]

            

            if len(proff_all_ans) == 0:

                ans_num[i] = 2

                last_day_activity[i] = 2

            else:

                #First factor

                ans_num[i] = self.sigmoid(len(proff_all_ans))



                #Second factor

                days = (datetime.datetime.now() - datetime.datetime.strptime(datetime.datetime.strptime(proff_all_ans[-1:]["answers_date_added"].values[0], "%Y-%m-%d %H:%M:%S UTC%z").strftime("%Y-%m-%d"), "%Y-%m-%d")).days



                if max_last_activity < days:

                    last_day_activity[i] = 2

                else:

                    last_day_activity[i] = self.sigmoid(np.sqrt(days))

            

            ans_email_match = self.email_proff[:][(self.email_proff["answers_author_id"] == i)]

            ans_email_match = ans_email_match[:][(ans_email_match["emails_recipient_id"] == i)]



            if len(ans_email_match) == 0:

                email_ind = email_ind[0]

                ans_email_num[i] = 1

                ans_email_time[i] = 1

            else:

                email_ind = email_ind[1]



                #Third factor

                ans_email_num[i] = self.sigmoid(len(ans_email_match))



                #Fourth factor

                ans_time = [] #Days between sending email and getting the answer on the question from email



                for index, row in ans_email_match.iterrows():

                    ans_time.append([row["emails_date_sent"], row["answers_date_added"]])



                mean = 0

                for item in ans_time:

                    email_send_date = datetime.datetime.strptime(datetime.datetime.strptime(item[0], "%Y-%m-%d %H:%M:%S UTC%z").strftime("%Y-%m-%d"), "%Y-%m-%d")

                    ans_made_date = datetime.datetime.strptime(datetime.datetime.strptime(item[1], "%Y-%m-%d %H:%M:%S UTC%z").strftime("%Y-%m-%d"), "%Y-%m-%d")



                    mean = mean + (ans_made_date - email_send_date).days



                mean = mean / len(ans_email_match)

                ans_email_time[i] = self.sigmoid(mean)



        #Get activity value for each candidate and sort them

        professionals = []

        for i in ans_email_num:

            activity_value = 1 + (ans_email_num[i] - ans_email_time[i]) * email_ind + ans_num[i] - last_day_activity[i]

            professionals.append([activity_value, i])



        professionals = sorted(professionals, key=lambda x: x[0]) 

        professionals.reverse()



        return professionals
num_of_prof = 10



model = Professionals(ans_proff, email_proff, num_of_prof=num_of_prof)



model.fit_for_tags(df) 

professionals = model.predict_by_tags('job')
professionals
professionals = np.array(professionals)
professionals_best = model.predict_by_activity(professionals[:num_of_prof, 1])
professionals_best