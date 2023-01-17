import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

from sklearn import tree
from nltk.tokenize import wordpunct_tokenize
train_df=pd.read_csv(r'../input/train.csv')

test_df=pd.read_csv(r'../input/test.csv')
test_df.head()
train_df.head()
import nltk

nltk.download("stopwords")
stop = set(stopwords.words('english'))
def common_word_percent(id,question1,question2):

    commonality_pc= {}

    for index in range(len(question1)):

        ques1_words = [i.lower() for i in wordpunct_tokenize(str(question1[index]))if i.lower() not in stop]

        ques2_words = [i.lower() for i in wordpunct_tokenize(str(question2[index])) if i.lower() not in stop]

        common_q1 = set(ques1_words).intersection( set(ques2_words) )

        common_q2 = set(ques2_words).intersection( set(ques1_words) )

        if (len(ques1_words)!=0 and len(ques2_words)!=0 and len(common_q2)!=0 and len(common_q1)!=0):

            commonality_pc[id[index]] = ((len(common_q1)+len(common_q2))/(len(ques1_words)+len(ques2_words)))*100

        else:

            commonality_pc[id[index]] = 0

    #print(commonality_pc)

    return commonality_pc      

    
train_df['common_words_pc']=common_word_percent(train_df.id.values,train_df.question1.values,train_df.question2.values).values()
train_df.head()
train_df.iloc[-1]
common_word_percent(train_df.id.values,train_df.question1.values,train_df.question2.values).values()
train_df.head()
train_df[train_df['is_duplicate']==1]
train_df[train_df['is_duplicate']==0]
plt.scatter(train_df[train_df['is_duplicate']==0].id,train_df[train_df['is_duplicate']==0].common_words_pc)
plt.scatter(train_df[train_df['is_duplicate']==1].id,train_df[train_df['is_duplicate']==1].common_words_pc)
train_df['is_duplicate'][train_df['common_words_pc']>30].value_counts(normalize=True)
test_df.head()
test_df['common_words_pc']=common_word_percent(test_df.test_id.values,test_df.question1.values,test_df.question2.values).values()
test_df.head()
from sklearn import tree
target = train_df['is_duplicate'].values

features_one = train_df[["common_words_pc"]].values
train_df.head()
my_tree_one = tree.DecisionTreeClassifier()

my_tree_one = my_tree_one.fit(features_one,target)
print(my_tree_one.feature_importances_)

print(my_tree_one.score(features_one,target))
test_features =test_df[["common_words_pc"]].values

my_prediction = my_tree_one.predict(test_features)

print(my_prediction)
QuestionId =np.array(test_df["test_id"]).astype(int)

my_solution = pd.DataFrame(my_prediction, QuestionId, columns = ["is_duplicate"])

print(my_solution)
from sklearn.ensemble import RandomForestClassifier





features_forest = train_df[["is_duplicate"]].values





forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(features_forest, target)





print(my_forest.score(features_forest, target))





test_features = test_df[["common_words_pc"]].values

pred_forest = my_forest.predict(test_features)

print(len(pred_forest))
features = train_df.iloc[:,0:5]

response = train_df.iloc[:,5]