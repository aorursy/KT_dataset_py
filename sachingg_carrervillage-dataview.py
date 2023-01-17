import numpy as np



import pandas as pd



import warnings



import os



import seaborn as sns



import matplotlib.pyplot as plt



import wordcloud



import matplotlib_venn





input_path = "../input"

for file in os.listdir(input_path):

    print(file)
professionals_df = pd.read_csv(r"../input/professionals.csv")

tag_users_df = pd.read_csv(r"../input/tag_users.csv")

school_memberships_df = pd.read_csv(r"../input/school_memberships.csv")

answers_df = pd.read_csv(r"../input/answers.csv")

tags_df = pd.read_csv(r"../input/tags.csv")

questions_df = pd.read_csv(r"../input/questions.csv")

students_df = pd.read_csv(r"../input/students.csv")

groups_df = pd.read_csv(r"../input/groups.csv")

group_memberships_df = pd.read_csv(r"../input/group_memberships.csv")

comments_df = pd.read_csv(r"../input/comments.csv")

matches_df = pd.read_csv(r"../input/matches.csv")

emails_df = pd.read_csv(r"../input/emails.csv")

tag_question_df = pd.read_csv(r"../input/tag_questions.csv")

answer_scores_df = pd.read_csv(r"../input/answer_scores.csv")

question_scores_df = pd.read_csv(r"../input/question_scores.csv")

print("professionals DF :",professionals_df.columns)

print("tag_users_df  :",tag_users_df.columns)

print("school_memberships_df DF :",school_memberships_df.columns)

print("answers_df DF :",answers_df.columns)

print("tags_df DF :",tags_df.columns)

print("questions_df DF :",questions_df.columns)

print("students_df DF :",students_df.columns)

print("groups_df DF :",groups_df.columns)

print("group_memberships_df DF :",group_memberships_df.columns)

print("comments_df DF :",comments_df.columns)

print("matches_df DF :",matches_df.columns)

print("emails_df DF :",emails_df.columns)

print("tag_question_df DF :",tag_question_df.columns)

print("answer_scores_df DF :",answer_scores_df.columns)

professionals_df.isna().sum()
def plotNa(df_list):

    tot = len(df_list)

    

    for dfi in df_list:

        f, axes = plt.subplots(2, figsize=(20, 10), sharex=True)

        count = 0

        df = dfi[0]

        df_name = dfi[1]

        sns.barplot(x=df.columns,y=df.isna().sum(),ax=axes[0])

        axes[0].set_title("Nans in the DataFrame " + df_name)

        sns.barplot(x=df.columns,y=df.nunique(),ax=axes[1])

        axes[1].set_title("Unique Values in DataFrame " + df_name)

        plt.plot()



def plotNaDf(df):

    x = df.columns

    y = df.isna().sum()

    f, axes = plt.subplots(2, figsize=(20, 10), sharex=True)



    s1 = sns.barplot(x=df.columns,y=df.isna().sum(),ax=axes[0])

    axes[0].set_title("Nans in the database")

    s2 = sns.barplot(x=df.columns,y=df.nunique(),ax=axes[1])

    axes[1].set_title("Unique Values in the database")

#Plots the venn diagram to see the common and uncommon col in 2 datasets

from matplotlib_venn import venn2, venn2_circles

def pltOnlyVenn(

    df1,

    df2,

    col1,

    col2,

    label1,

    label2,

    title1):



    f,ax1 = plt.subplots(figsize=(5, 5))

    venn2(subsets=(set(df1[col1]),set(df2[col2])),

          set_labels=[label1,label2],ax=ax1)

    ax1.set_title(title1)

    plt.tight_layout()

    return



def pltVenn(

    df1,

    df2,

    col1,

    col2,

    label1,

    label2,

    title1,

    pieLabel1,

    pieLabel2,

    pieLabel3,

    pieTitle):

    

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    #f, axes = plt.subplots(2,figsize=(10, 10), sharex=True)

    venn2(subsets=(set(df1[col1]),set(df2[col2])),set_labels=[label1,label2],ax=ax1)

    ax1.set_title(title1)



    #Pie

    ax2.set_title(pieTitle)

    ax2.pie(

                x = [

                    len(set(df1[col1]) - set(df2[col2])),

                    len(set(df1[col1]) & set(df2[col2])),

                    len(set(df2[col2]) - set(df1[col1]))

                ],

        labels=[pieLabel1,pieLabel2,pieLabel3],

        autopct='%1.2f%%'

    )

    plt.tight_layout()



def pltPie(

    df1,

    df2,

    col1,

    col2,

    pieLabel1,

    pieLabel2,

    pieTitle):

    

    f, ax2 = plt.subplots(figsize=(5, 5))



    #Pie

    ax2.set_title(pieTitle)

    ax2.pie(

                x = [

                    len(set(df1[col1]) - set(df2[col2])),

                    len(set(df1[col1]) & set(df2[col2]))

                ],

        labels=[pieLabel1,pieLabel2],

        autopct='%1.2f%%'

    )

    plt.tight_layout()



pltOnlyVenn(students_df,

           professionals_df,

           'students_id',

           'professionals_id',

           'students_df',

           'professionals_df',

           'Common ID within students and professionals')
group_memberships_df.head()
pltVenn(

    students_df,

    questions_df,

    'students_id',

    'questions_author_id',

    'Students',

    'Questions',

    'How many students have been writing questions',

    'Students without questions',

    'Students with questions',

    'Not students',

    'What is the percentage'

)

pltVenn(

    professionals_df,

    questions_df,

    'professionals_id',

    'questions_author_id',

    'Professionals',

    'Questions',

    'How many Professionals have been writing questions',

    'Professionals without questions',

    'Professionals with questions',

    'Not Professionals',

    'What is the percentage'

)

pltVenn(

        answers_df,

        professionals_df,

        'answers_author_id',

        'professionals_id',

        'Students_df',

        'Professional_df',

        'authorID in students Vs professionals_id',

        'Professionals without answers',

        'Professionals with answers',

        'Not Professionals',

        'What is the percentage'

       )
#lets check the various group type

groups_df.groups_group_type.unique()
#Lets check if there are any common values in group_memberships_df & groups_df

pltVenn(

            groups_df,

            group_memberships_df,

            'groups_id',

            'group_memberships_group_id',

            'groups_df',

            'group_memberships_df',

            'Common Group Ids',

            'Unique in groups_df',

            'Common Across',

            'Unique in group_memberships_df',

            'What is the Percentage')
group_memberships_df['groups_id'] = group_memberships_df['group_memberships_group_id']

diff_groups_df = groups_df.merge(group_memberships_df,

                          on='groups_id',how='right')



diff_groups_df.groups_group_type.unique()
pltVenn(

            students_df,

            questions_df,

            'students_id',

            'questions_author_id',

            'students_df',

            'questions_df',

            'Students Asking Questions',

            'Students Not Asking Questions',

            'Students Asking Questions',

            'Unique Students in questions_df',

            'What is the Percentage')
pltVenn(

            professionals_df,

            comments_df,

            'professionals_id',

            'comments_author_id',

            'professionals_df',

            'comments_df',

            'Professional Giving Comments',

            'Professional Not Giving Comments',

            'Professional Giving Comments',

            'Unique Professional in comments_df',

            'What is the Percentage')
pltVenn(

            students_df,

            comments_df,

            'students_id',

            'comments_author_id',

            'students_df',

            'comments_df',

            'Students Giving Comments',

            'Students Not Giving Comments',

            'Students Giving Comments',

            'Unique Students in comments_df',

            'What is the Percentage')
answers_df['professionals_id'] = answers_df['answers_author_id']



ans_prof_df = answers_df.merge(professionals_df,

                          on='professionals_id',how='left')

ans_prof_df.shape

ans_prof_df.head()
answer_scores_df['answers_id'] = answer_scores_df['id']

score_ans_df = answer_scores_df.merge(answers_df,on='answers_id',how='left')

print(score_ans_df.shape)
def getDataWithColumns(df):

    allData = []

    for i in df:

        df_name = i[1]

        columns = i[0].columns.values

        allData.append((df_name,columns))

    allData_df = pd.DataFrame(allData,columns=['DataFrame', 'Columns'])

    return allData_df



def getDateTimeMonth(df,col):

    df[col] = df[col].astype('datetime64')

    df['year'] = df[col].dt.year

    df['month'] = df[col].dt.month

    df['week'] = df[col].dt.week

    df['day'] = df[col].dt.day

    df['hour'] = df[col].dt.hour

    return df
professionals_df = getDateTimeMonth(professionals_df,col='professionals_date_joined')
getDateTimeMonth(comments_df,col='comments_date_added')

getDateTimeMonth(questions_df,col='questions_date_added')

getDateTimeMonth(students_df,col='students_date_joined')
plt.figure(figsize=(10,10))

plt.plot(professionals_df.year)

plt.plot(students_df.year)
plt.figure(figsize=(10,10))

plt.plot(question_scores_df.score)
emails_df['professionals_id'] = emails_df['emails_recipient_id']

email_prof_df = emails_df.merge(professionals_df,on='professionals_id',how='left')

email_prof_df.shape
pltVenn(matches_df,

       emails_df,

       'matches_email_id',

       'emails_id',

       'matches_df',

       'emails_df',

        'Matching Answers',

        'Emails in Matching',

        'Common Emails',

        'emails in emails_df & not in matches_df',

        'Pie Chart for Matching emails'

       )
pltPie(

    emails_df,

    matches_df,

    'emails_id',

    'matches_email_id',

    'emails_df  - matches_df',

    'emails_df  & matches_df',

    'Pie Chart for Matches Email')

emails_df.emails_id.nunique() / matches_df.matches_email_id.nunique()
email_match = matches_df.matches_email_id.unique()

email_match.shape
email_prof_df.head(1)
prof_email_match = email_prof_df.loc[email_prof_df['emails_id'].isin(email_match)]



uniq_prof_id = prof_email_match.professionals_id.unique()
uniq_prof_id.shape
uniqueProf = professionals_df[professionals_df['professionals_id'].isin(uniq_prof_id)]
uniqueProf.shape
professionals_df.shape
pltPie(

professionals_df,

uniqueProf,

'professionals_id',

'professionals_id',

'professionals_df - uniqueProf',

'professionals_df & uniqueProf',

'PieChart for Professors who answer correctly')