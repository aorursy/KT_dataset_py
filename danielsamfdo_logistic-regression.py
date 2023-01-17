# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from sklearn.linear_model import LogisticRegression





# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')

names = ['Age', 'Gender', 'DayOfTheWeek', 'Diabetes', 'Alcoolism', 'HiperTension', 'Handcap', 'Smokes', 'Scholarship', 'Tuberculosis', 'Sms_Reminder', 'AwaitingTime', 'Status']

data_frame = pd.concat([df['Age'], df['Gender'], df['DayOfTheWeek'], df['Diabetes'],  df['Alcoolism'], df['HiperTension'], df['Handcap'], df['Smokes'], df['Scholarship'], df['Tuberculosis'], df['Sms_Reminder'], df['AwaitingTime'], df['Status']], axis=1, keys=names)

di = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday" : 5, "Saturday" : 6, "Sunday" : 7}

data_frame.replace({"DayOfTheWeek": di}, inplace=True)

data_frame.replace({"Gender": {"M": 0, "F":1}}, inplace=True)

data_frame.replace({"Status": {"Show-Up": 1, "No-Show":0}}, inplace=True)

data_frame
from sklearn.utils import shuffle

data_frame = shuffle(data_frame)

def print_accuracy_fscore(Y_pred, Y_test):

  count = 0.0

  correct = 0.0

  scores = {'tp':0.0,'tn':0.0,'fp':0.0,'fn':0.0}



  for i in range(len(Y_pred)):

      count += 1

      if Y_pred[i] == Y_test[i]:

          correct += 1 



      #precision and recall

      #true positive 

      if Y_test[i] == 1:

          if Y_pred[i] == 1:

              scores['tp'] += 1

          else:

              scores['fn'] += 1

      else:

          if Y_pred[i] == 1:

              scores['fp'] += 1

          else:

              scores['tn'] += 1

  

  print(scores)

  test_acc = correct / count 

  if(scores['tp']==0):

    precision = 0

    recall = 0

    f_score = 0

  else:

    precision = scores['tp'] / (scores['tp'] + scores['fp'])

    recall = scores['tp'] / (scores['tp'] + scores['fn'])

    f_score = (2*precision*recall)/(precision+recall)

  print("Precision : " + str(precision))

  

  print("Recall : "+ str(recall))

  

  print("F score : " + str(f_score))

  

  print("Accuracy : "+ str(test_acc))





def count(result):

  counts = dict()

  for i in result:

    counts[i] = counts.get(i, 0) + 1

  print(counts)
def classify(train_data_frame, train_output, test_data_frame, test_output):

    classifier = LogisticRegression(C=1e5)

    classifier = classifier.fit(train_data_frame, train_output )

    result = classifier.predict(train_data_frame)

    #print result

    #print len(result)

    count(result)

    print_accuracy_fscore(result, list(train_output['Status']))



    result = classifier.predict(test_data_frame)

    print_accuracy_fscore(result, list(test_output['Status']))

    count(result)
msk = data_frame['Status'] == 1

NoShow = data_frame[~msk]

Show = data_frame[msk]





length_train = 50000

train_show = (Show[:length_train])

test_show = (Show[length_train:])



train_noshow = (NoShow[:length_train])

test_noshow = (NoShow[length_train:])



train_show_output = (Show['Status'][:length_train])

test_show_output = (Show['Status'][length_train:])

train_noshow_output = (NoShow['Status'][:length_train])

test_noshow_output = (NoShow['Status'][length_train:])



train_df = pd.concat([train_show, train_noshow], ignore_index=True)

train_op_df = pd.concat([train_show_output, train_noshow_output], ignore_index=True)

test_df = pd.concat([test_show, test_noshow], ignore_index=True)

test_op_df = pd.concat([test_show_output, test_noshow_output], ignore_index=True)

train_op_df=pd.DataFrame(train_op_df, columns = ["Status"])

test_op_df=pd.DataFrame(test_op_df, columns = ["Status"])

classify(train_df, train_op_df, test_df, test_op_df)