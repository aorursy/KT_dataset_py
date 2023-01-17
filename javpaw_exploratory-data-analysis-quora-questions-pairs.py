# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
pd.set_option('display.max_colwidth', 200)
# Any results you write to the current directory are saved as output.
class Lodash:    
    def flow(self, *args):
        def fns(payload):
            result = payload
            for fn in args:
                result = fn(result)
            return result
        return fns
dash = Lodash()
train  =  pd.read_csv('../input/train.csv')
train.head(10)
def getNumberOfRows(df):
    return len(df)

def getDuplicates(df):
    return df[df['is_duplicate'] == 1]

def getDuplicatesPercentage(df):
    duplicateRows = dash.flow(getDuplicates, getNumberOfRows)(df)
    totalRows = getNumberOfRows(df)
    return round(duplicateRows*100/totalRows,2)

def getQuestionsIds(df):
    return df['qid1'].tolist() + df['qid2'].tolist()

def getTotalNumberOfQuestions(df):
    return dash.flow(getQuestionsIds, len)(df)

def getUniqueQuestionsIds(df):
    return dash.flow(getQuestionsIds, set)(df)

def getTotalNumberOfUniqueQUestions(df):
    return dash.flow(getUniqueQuestionsIds, len)(df)

def getRepeatedQuestionsIds(df):
    serie = dash.flow(getQuestionsIds, pd.Series)(df)
    counts = serie.value_counts()
    repeated = counts[counts > 1]
    return list(set(repeated.index.tolist()))

def getNumberOfRepeatedQuestionsIds(df):
    return dash.flow(getRepeatedQuestionsIds, len)(df)

def getPercentageOfRepeatedQuestions(df):
    return round(getNumberOfRepeatedQuestionsIds(df)/getTotalNumberOfUniqueQUestions(df),2)*100
def printReport(df):
    print("Total Number of Questions Pairs: {:,}".format(getNumberOfRows(df)))
    print("Total Number of Questions: {:,}".format(getTotalNumberOfQuestions(df)))
    print("Total Number of Unique Questions: {:,}". format(getTotalNumberOfUniqueQUestions(df)))
    print("Total Number of Repeated Questions: {:,}".format(getNumberOfRepeatedQuestionsIds(df)))
    print("Percentage of Repeated Questions: {:,}%".format(getPercentageOfRepeatedQuestions(df)))
    print('\n')

printReport(train)
numberOfQuestions = train.size
numberOfDuplicatedPairs = train[train['is_duplicate'] == 1].size
percentageDuplicatedPairs = numberOfDuplicatedPairs*100/numberOfQuestions
# print('Total number of question pairs for training: {}'.format(len(df_train)))
# print('Duplicate pairs: {}%'.format(round(df_train['is_duplicate'].mean()*100, 2)))
# qids = pd.Series(df_train['qid1'].tolist() + df_train['qid2'].tolist())
# print('Total number of questions in the training data: {}'.format(len(
#     np.unique(qids))))
# print('Number of questions that appear multiple times: {}'.format(np.sum(qids.value_counts() > 1)))
print('The size of the training set is: {}'.format(numberOfQuestions))
print('Total number of duplicated pairs: {}'.format(numberOfDuplicatedPairs))
print('Percentage Duplicated pairs: {}'.format(percentageDuplicatedPairs))
count_duplicate = train['is_duplicate'].value_counts()
plt.bar([0, 1], count_duplicate.values)
plt.ylabel('Number of pairs')
plt.xlabel('Duplicate', fontsize=12)
plt.title('Balance of classes in training set')
plt.show()


questions_train = pd.concat([train['question1'], train['question2']])
q_train_len = questions_train.apply(lambda x: len(str(x).split(' ')))
quantiles_train = q_train_len.quantile([0.25, 0.5, 0.75, 0.99])
print("Quantiles Train: ")
print(quantiles_train)
plt.figure(figsize=(10, 5))
plt.hist(q_train_len, bins=100, range=[0,300])
plt.title('Length Training set')
plt.xlabel('Number of characters per question')
plt.ylabel('Number of questions')
plt.yscale('log', nonposy='clip')
test  =  pd.read_csv('../input/test.csv')
test.head(10)
questions_test = pd.concat([test['question1'], test['question2']])
q_test_len = questions_test.apply(lambda x: len(str(x).split(' ')))
quantiles_test = q_test_len.quantile([0.25, 0.5, 0.75, 0.99])
print("Quantiles test: ")
print(quantiles_test)
plt.figure(figsize=(10, 5))
plt.hist(q_test_len, bins=100, range=[0,300])
plt.title('Length Training set')
plt.xlabel('Number of characters per question')
plt.ylabel('Number of questions')
plt.yscale('log', nonposy='clip')

def getRowsWithNoRepeatedQuestion(df):
    nonRepeatedQeuestions = set(getUniqueQuestionsIds(df)) - set(getRepeatedQuestionsIds(df))
    return df[ (df['qid1'].isin(nonRepeatedQeuestions) & df['qid2'].isin(nonRepeatedQeuestions)) ]

def getRowsWithAtLeastOneRepeatedQuestion(df):
    repeatedQuestionsIds = getRepeatedQuestionsIds(df)
    nonRepeatedQeuestions = set(getUniqueQuestionsIds(df)) - set(getRepeatedQuestionsIds(df))
    onlyQ1IsRepeated = df['qid1'].isin(repeatedQuestionsIds) & df['qid2'].isin(nonRepeatedQeuestions)
    onlyQ2IsRepeated = df['qid2'].isin(repeatedQuestionsIds) & df['qid1'].isin(nonRepeatedQeuestions)

    return df[ onlyQ1IsRepeated | onlyQ2IsRepeated]

def getRowsWithTwoRepeatedQuestions(df):
    repeatedQuestionsIds = getRepeatedQuestionsIds(df)
    return df[ (df['qid1'].isin(repeatedQuestionsIds) & df['qid2'].isin(repeatedQuestionsIds)) ]

    
def printDuplicatePairsReport(df):
    print("Total Number of Questions Pairs: {:,}".format(getNumberOfRows(df)))
    print("Percentage of pairs Marked as duplicate: {:,}%\n".format(getDuplicatesPercentage(df)))

def printStatisticsForPairsWithRepeatedQuestions(df):
    print("## Statistics for rows with no repeated questions ##")
    dash.flow(getRowsWithNoRepeatedQuestion, printDuplicatePairsReport)(df)
    print("## Statistics for rows with one repeated question ##")
    dash.flow(getRowsWithAtLeastOneRepeatedQuestion, printDuplicatePairsReport)(df)
    print("## Statistics for rows with two repeated questions ##")
    dash.flow(getRowsWithTwoRepeatedQuestions, printDuplicatePairsReport)(df)

printStatisticsForPairsWithRepeatedQuestions(train)
#Read the file
#test = pd.read_csv('../input/test.csv', nrows= 10000)
test = pd.read_csv('../input/train.csv', nrows = 100)

duplicate_score = test['is_duplicate'].mean()
def classifyByRepetition(row, repeatedQuestions):
    q1IsRepeated = row['question1'] in repeatedQuestions
    q2IsRepeated = row['question2'] in repeatedQuestions
    
    if not q1IsRepeated and not q2IsRepeated:
        return 0.224
    elif q1IsRepeated and q2IsRepeated:
        return 0.0716
    else:
        return 0.1571

def getRepeatedQuestions(df):
    return set(df['question1'].append(df['question2'], ignore_index = True)
        .value_counts()
        .where(lambda x: x>1)
        .dropna()
        .index
        .get_values())
    
def addIsDuplicatedColum(df):
    getIsDuplicateValue = lambda row: classifyByRepetition(row, getRepeatedQuestions(df))
    df['prediction_is_duplicate'] =  df.apply(getIsDuplicateValue, axis=1 )
    return df

def calculate_and_print_benchmark_based_on_duplicate_percentage(df):
    print('Binary Cross-Entropy Score based on duplicates:', log_loss(df['is_duplicate'], np.zeros_like(df['is_duplicate']) + duplicate_score)) 

def calculate_and_print_benchmark_based_on_repetition(df):
    print('Binary Cross-Entropy Score based on questions repetition:', log_loss(df['is_duplicate'], df['prediction_is_duplicate']))
    
def mainAddIsDuplicatedColumnAndTransformToCsv(df):
    df = addIsDuplicatedColum(df)
    calculate_and_print_benchmark_based_on_duplicate_percentage(df)
    calculate_and_print_benchmark_based_on_repetition(df)
    #df.to_csv('base_submission.csv', index = False, columns=['test_id', 'is_duplicate'])

mainAddIsDuplicatedColumnAndTransformToCsv(test)



def isOpenQuestion(q):
    yesNoQuestionsInitializers = ['is', 'are', 'should', 'do', 'does', 'can']
    openQuestionInitializer = ['what', 'how', 'why', 'who', 'when', 'where', 'which', "what's", "how's", "why's", "who's", "when's", "where's"]
    isOpen = any( str(q).lower().startswith(i) for i in openQuestionInitializer)
    isYesNo = any( str(q).lower().startswith(i) for i in yesNoQuestionsInitializers)
    if isOpen:
        return 0
    elif isYesNo:
        return 1
    else:
        return 2

train['Q1TypeOfQuestion'] = train['question1'].apply(isOpenQuestion)
train['Q2TypeOfQuestion'] = train['question2'].apply(isOpenQuestion)
numberOfYesNoQuestions = train[train['Q1TypeOfQuestion'] == 1].shape[0] + train[train['Q2TypeOfQuestion'] == 1].shape[0]
print("Number of yes/no questions: ", numberOfYesNoQuestions)
numberOfOpenQuestions = train[train['Q1TypeOfQuestion'] == 0].shape[0] + train[train['Q2TypeOfQuestion'] == 0].shape[0]
numberOfOpenQuestions
print("Number of Open questions: ", numberOfOpenQuestions)
numberOfNonClassifiedQuestions = train[train['Q1TypeOfQuestion'] == 2].shape[0] + train[train['Q2TypeOfQuestion'] == 2].shape[0]
numberOfNonClassifiedQuestions
print("Number of no classified questions: ", numberOfNonClassifiedQuestions)
corrMat = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(0,3):
    for j in range(0,3):
        corrMat[i][j] = train[(train['Q1TypeOfQuestion'] == i) & (train['Q2TypeOfQuestion'] == j)].is_duplicate.mean()
corrMat = np.array(corrMat)
sns.heatmap(corrMat, vmax=0.5, square=True, annot=True)