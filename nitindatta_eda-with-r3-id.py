import pandas as pd

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

sns.set_style('dark')
df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows=100000)

ex_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
df.head(10).T
df.describe().T
df.isnull().sum()
df.info()
cols = df.columns

for col in cols: 

    print(f'Unique values in    {col} :{df[col].nunique()}')
sns.set_style('whitegrid')

fig,ax=plt.subplots(figsize=(18,12))

plt.subplot(2, 2, 1)

g1=sns.distplot(df['user_id'],color='orange',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 2, "label": "KDE"})

g1.set_title("User_ID")



plt.subplot(2, 2, 2)

g2=sns.distplot(df['content_id'],color='red',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 2, "label": "KDE"})

g2.set_title("Content_ID")



plt.subplot(2, 2, 3)



g3=sns.distplot(df['task_container_id'],color='cyan',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 2, "label": "KDE"})

g3.set_title("Task_container_ID")



plt.subplot(2, 2, 4)



g3=sns.distplot(df['prior_question_elapsed_time'],color='blue',hist_kws={'alpha':1,"linewidth": 2}, kde_kws={"color": "k", "lw": 2, "label": "KDE"})

g3.set_title("Prior_question_elapsed_time")



plt.figure(figsize=(10,6))

sns.set_style('dark')

df['timestamp'].hist(bins = 50,color='yellow')
corr_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', nrows=2000000)

corr = corr_df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(10, 10))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="viridis",annot=True)
correct_counts = df['answered_correctly'].value_counts().reset_index()

correct_counts.columns = ['answered_correctly','per']

correct_counts['per'] /= len(df)

correct_counts = correct_counts[correct_counts.answered_correctly != -1]





colors = ('red','black')



explode = (0.1,0.1)



wp = {'linewidth':1, 'edgecolor':'black'}



# Creating autocpt arguments 

def func(pct, allvalues): 

    absolute = int(pct / 100.*np.sum(allvalues)) 

    return "{:.1f}%".format(pct) 



fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,22))

wedges, texts, autotexts = ax1.pie(correct_counts['per'],  

                                  autopct = lambda pct: func(pct, correct_counts['per']), 

                                  explode = explode,  

                                  labels = correct_counts['answered_correctly'], 

                                  shadow = True, 

                                  colors = colors, 

                                  startangle = 90, 

                                  wedgeprops = wp, 

                                  textprops = dict(color ="white")) 



ax1.legend(wedges, correct_counts['answered_correctly'], 

          title ="Answered_correctly", 

          loc ="center", 

          bbox_to_anchor =(1, 0, 0, 0)) 



plt.setp(autotexts, size = 14, weight ="bold") 

ax1.set_title("Answered Correctly percentage") 

  



# ############################################################

user_counts = df['user_answer'].value_counts().reset_index()

user_counts.columns = ['user_answer','per']

user_counts['per'] /= len(df)

user_counts = user_counts[user_counts.user_answer != -1]



colors1 = ('orange','red','brown','black')



explode1 = (0.1,0.1,0.1,0.1)



wp = {'linewidth':1, 'edgecolor':'black'}



wedges, texts, autotexts = ax2.pie(user_counts['per'],  

                                  autopct = lambda pct: func(pct, user_counts['per']), 

                                  explode = explode1,  

                                  labels = user_counts['user_answer'], 

                                  shadow = True, 

                                  colors = colors1, 

                                  startangle = 90, 

                                  wedgeprops = wp, 

                                  textprops = dict(color ="white")) 



ax2.legend(wedges, user_counts['user_answer'], 

          title ="User Answeres", 

          loc ="center left", 

          bbox_to_anchor =(1, 0, 0, 0)) 



plt.setp(autotexts, size = 14, weight ="bold") 

ax2.set_title("User Answer percentage") 

  

# show plot 

plt.show() 
plt.figure(figsize=(20,12))

sns.set_style('dark')



mini_df= df.copy()

mini_df = mini_df.sort_values(by=['timestamp'])

mini_df = mini_df.drop_duplicates('timestamp')



# Start

min_df = mini_df.head(100)

plt.subplot(3, 1, 1)

sns.pointplot(x=min_df['timestamp'],y=min_df['prior_question_had_explanation'],hue= min_df['answered_correctly'],

              linestyle='--',color='yellow',markers='x')

plt.title('Start_time')

plt.xticks([])

plt.yticks([-0.05,0,1])



# Mid

mid_df = mini_df[50000:51100]

plt.subplot(3, 1, 2)

sns.pointplot(x=mid_df['timestamp'],y=mid_df['prior_question_had_explanation'],hue= mid_df['answered_correctly'],

              linestyle='--',color='orange',markers='x')

plt.title('Middle_time')

plt.xticks([])

plt.yticks([0,1])



# End

max_df = mini_df.tail(100)

plt.subplot(3, 1, 3)

sns.pointplot(x=max_df['timestamp'],y=max_df['prior_question_had_explanation'],hue= max_df['answered_correctly'], 

              linestyle='--',color='red',markers='x')

plt.title('End_time')

plt.xticks([])

plt.yticks([0,1])
df = df[df.user_answer != -1]

def annotate(data, **kws):

    n = len(data)

    ax = plt.gca()

    ax.text(.7, .8, f"N = {n}", transform=ax.transAxes)

sns.set_style('whitegrid')

g = sns.FacetGrid(df, col="user_answer", height=4.5, aspect=0.8)

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)



g.map(sns.scatterplot, 'timestamp', 'prior_question_elapsed_time',alpha=1, edgecolor='black',color='red')

g.map_dataframe(annotate)

fig,ax=plt.subplots(figsize=(18,6))

sns.set_style('dark')

plt.subplot(1, 2, 1)

g1=sns.countplot(df['user_answer'],palette='rocket', hue = df['prior_question_had_explanation'],**{'hatch':'/','linewidth':3})

g1.set_title("User Answer")



plt.subplot(1, 2, 2)

g2=sns.countplot(df['answered_correctly'],palette='rocket',hue= df['prior_question_had_explanation'],**{'hatch':'/','linewidth':3})

g2.set_title("Answered Correctly")
y_cols = ['user_id','content_id','task_container_id',]

x_cols = 'prior_question_elapsed_time'

sns.set_style('white')

plt.figure(figsize=(18,6))

for i in range(len(y_cols)):

    plt.subplot(1, 3, i+1)

    sns.scatterplot(x=x_cols,y=y_cols[i],data=df, alpha=0.8, color='yellow',edgecolor="r")

    plt.title(y_cols[i],fontsize=10)
plt.figure(figsize=(10,6))

sns.set_style('darkgrid')

sns.scatterplot(x = df['task_container_id'], y= df['prior_question_elapsed_time'], hue=df['user_id'],palette='plasma',linewidth=0, size=df['user_id'] ,alpha=1)
sns.set_style('white')

plt.figure(figsize=(10,6))

sns.set_style('whitegrid')

sns.scatterplot(x ='timestamp', y='prior_question_elapsed_time', data = df, hue='prior_question_had_explanation',alpha=0.8

                ,linewidth=0,palette='viridis')
plt.figure(figsize=(10,6))

sns.countplot(df['user_answer'], hue=df['answered_correctly'],palette='Set3',**{'hatch':'-','linewidth':1})

plt.title('User_Answer vs Correctness', fontsize = 20)

plt.show()
plt.figure(figsize=(10,6))

sns.set_style('darkgrid')

sns.scatterplot(data=df, x ='content_id', y='user_id',hue='user_id', size='user_id',palette='plasma_r',linewidth=0,alpha=1)
min_df = df.groupby('user_id').agg({'answered_correctly': 'sum', 'row_id':'count'})

sns.set_style('darkgrid')

plt.figure(figsize = (16,8))

sns.distplot((min_df['answered_correctly'] * 100)/min_df['row_id'],color='orange',hist_kws={'alpha':1,"linewidth": 4}, kde_kws={"color": "black", "lw": 2, "label": "KDE"})

plt.title('Distribution of correct answers percentage by each user', fontdict = {'size': 12})

plt.xlabel('Percentage of correct answers', size = 12)
lec = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')
lec.head(8).T
lec.describe().T
lec.info()
cols = lec.columns

for col in cols: 

    print(f'Unique values in    {col} : {lec[col].nunique()}')
sns.set_style('dark')

sns.pairplot(data=lec,hue='type_of',markers=["^", "o", "v",'h'], corner=True,plot_kws=dict(linewidth=0, alpha=1),height=4,palette=['red','blue','black','yellow'])
sns.set_style('white')

plt.figure(figsize=(8,8))

sns.scatterplot(data=lec, x='lecture_id', y='tag', hue='part', size='part')
sns.set_style('dark')

plt.figure(figsize=(8,8))

sns.scatterplot(lec['tag'],lec['part'],hue=lec['type_of'], style=lec['type_of'],linewidth=0, palette='spring', alpha=1)
plt.figure(figsize=(15,12))



sns.set_style('darkgrid')

plt.subplot(221)

sns.countplot(lec['part'], palette='spring')

plt.title('Parts count',color='red')



sns.set_style('whitegrid')

plt.subplot(222)

sns.boxenplot(y=lec['part'],x=lec['type_of'],palette='spring_r')

plt.title('Part vs type_of lecture - BOXENPLOT', color='red')



sns.set_style('darkgrid')

plt.subplot(223)

sns.countplot(lec['type_of'], palette='autumn_r')

plt.title('Type of lectures count',color='orange')



sns.set_style('whitegrid')

plt.subplot(224)

sns.pointplot(y=lec['part'], x=lec['type_of'], color='red')

plt.title('Part vs type_of lecture - POINTPLOT',color='orange')
ques = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')

ques.head().T
ques.describe().T
import missingno as msno

msno.bar(ques,(8,6),color='red')

plt.title('MISSING VALUES',fontsize=14)
print('Before dropping shape', ques.shape)

ques = ques.dropna()

print('Afterr dropping shape', ques.shape)
sns.set_style('dark')

sns.pairplot(ques, corner=True,plot_kws=dict(linewidth=0, alpha=1,color='orange'),height=3, diag_kws={'edgecolor':'red','fill':True, 'color':'yellow'})



plt.figure(figsize=(10,6))

sns.set_style('whitegrid')

sns.scatterplot(x=ques['question_id'], y=ques['part'], hue=ques['correct_answer'],palette='viridis',linewidth=0 ,alpha=1,size=ques['correct_answer'])
plt.figure(figsize=(10,6))

sns.set_style('whitegrid')

sns.scatterplot(x=ques['bundle_id'], y=ques['part'], hue=ques['correct_answer'],palette='viridis',linewidth=0 ,alpha=1,size=ques['correct_answer'])
print("Is question_id column equal to bundle_id column?\nAnswer: ",ques['bundle_id'].equals(ques['question_id']))
ex_test = pd.read_csv('../input/riiid-test-answer-prediction/example_test.csv')
ex_test.head(8).T