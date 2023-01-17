import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/ufcdata/data.csv')
#Let's change the 'date' from object to date

df['date'] = pd.to_datetime(df['date'])



blue_wins = sum(df['Winner'] == 'Blue')

red_wins = sum(df['Winner'] == 'Red')



df_recent = df.loc[df['date'] >'01/03/2010']

df_old = df.loc[df['date'] <'01/04/2010']



blue_wins_recent = sum(df_recent['Winner'] == 'Blue')

red_wins_recent = sum(df_recent['Winner'] == 'Red')

blue_wins_old = sum(df_old['Winner'] == 'Blue')

red_wins_old = sum(df_old['Winner'] == 'Red')
x_labels = ('Blue', 'Red')

y_pos = np.arange(len(x_labels))

wins = ((blue_wins / (blue_wins + red_wins))*100, (red_wins / (blue_wins + red_wins))*100)

plt.bar(y_pos, wins, align='center', edgecolor=['blue', 'red'], color='lightgrey')

plt.xticks(y_pos, x_labels)

plt.title("Winning Percentage (Whole Dataset)")

plt.ylabel("Percent")
x_labels = ('Blue', 'Red')

y_pos = np.arange(len(x_labels))

wins = (blue_wins_old, red_wins_old)

plt.bar(y_pos, wins, align='center', edgecolor=['blue', 'red'], color='lightgrey')

plt.xticks(y_pos, x_labels)

plt.title("Total Wins (Prior to 1/4/2010)")

plt.ylabel("# of Wins")
x_labels = ('Blue', 'Red') 

y_pos = np.arange(len(x_labels)) 

wins = ((blue_wins_recent / (blue_wins_recent + red_wins_recent))*100, (red_wins_recent / (blue_wins_recent + red_wins_recent))*100) 

plt.bar(y_pos, wins, align='center', edgecolor=['blue', 'red'], color='lightgrey') 

plt.xticks(y_pos, x_labels)

plt.title("Winning Percentage (After 1/3/2010)")

plt.ylabel("Percent")
missing=(df.isnull().sum() / len(df)) * 100

missing = pd.DataFrame({'missing-ratio' :missing})

missing['feature'] = missing.index

over_10 = missing[missing['missing-ratio'] > 10]



remove_features = over_10['feature'].tolist()



#Remove features with over 10% missing

df_filtered = df.drop(remove_features, axis=1)
df_filtered.shape
f_missing=(df_filtered.isnull().sum() / len(df_filtered)) * 100

f_missing = pd.DataFrame({'missing-ratio' :f_missing})

f_missing['feature'] = f_missing.index

f_missing = f_missing[f_missing['missing-ratio'] > 0]

display(f_missing)

#Here are the other features we need to deal with.  We are going to do a mix of filling in averages or

#the most common value depending on what makes the most sense
df_filtered['Referee'] = df_filtered['Referee'].fillna('Unknown')

df_filtered['B_Stance'] = df_filtered['B_Stance'].fillna('Orthodox')

df_filtered['R_Stance'] = df_filtered['R_Stance'].fillna('Orthodox')

df_filtered['R_Height_cms'] = df_filtered['R_Height_cms'].fillna((df_filtered['R_Height_cms'].mean()))

df_filtered['B_Height_cms'] = df_filtered['B_Height_cms'].fillna((df_filtered['B_Height_cms'].mean()))

df_filtered['B_Weight_lbs'] = df_filtered['B_Weight_lbs'].fillna((df_filtered['B_Weight_lbs'].mean()))

df_filtered['R_Reach_cms'] = df_filtered['R_Reach_cms'].fillna((df_filtered['R_Reach_cms'].mean()))

df_filtered['R_Weight_lbs'] = df_filtered['R_Weight_lbs'].fillna((df_filtered['R_Weight_lbs'].mean()))

df_filtered['B_age'] = df_filtered['B_age'].fillna((df_filtered['B_age'].mean()))

df_filtered['R_age'] = df_filtered['R_age'].fillna((df_filtered['R_age'].mean()))
#To keep the number of features manageable after dummification we are going to remove B_fighter, R_fighter,

#and Referee.  Draws also needs to be removed.



to_drop = ['R_fighter', 'B_fighter', 'Referee']

df_filtered.drop(to_drop, axis=1, inplace=True)

df_filtered = df_filtered[df_filtered.Winner != 'Draw']
df_filtered
#Set the label column

df_filtered["Winner"] = df_filtered["Winner"].astype('category')

df_filtered["label"] = df_filtered["Winner"].cat.codes
df_total = df_filtered

y_total = df_total["label"]
#Let's make the test_set

df_test = df_total.loc[df_total['date'] >'11/11/2018']

y_test = df_test['label']



#Let's make the total train set

total_train = df_total.loc[df_total['date'] <'11/12/2018']

total_train_y = total_train['label']



#Let's make the recent train set

recent_train = total_train.loc[total_train['date']>'01/03/2010']

recent_train_y = recent_train['label']
#Lets remove the date, winner, and label from the training sets....

final_drop = ['date', 'Winner', 'label']

df_test.drop(final_drop, axis=1, inplace=True)

total_train.drop(final_drop, axis=1, inplace=True)

recent_train.drop(final_drop, axis=1, inplace=True)
#Takes a training set, training labels, test set, test labels, and a model.  Returns some

#stats and visualization



def run_model(X_train, y_train, X_test, y_test, model):

    #dummify and model

    X_test = pd.get_dummies(X_test)

    X_train = pd.get_dummies(X_train)

    X_train, X_test = X_train.align(X_test, join='left', axis=1)

    X_test.fillna(value=0, inplace=True)

    X_train.fillna(value=0, inplace=True)

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)

    X_test = sc.transform(X_test)    

    

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    class_names = ['Blue', 'Red']

    

    titles_options= [(f"Total Confusion matrix", None),

                     ("Normalized confusion matrix", 'true')]

    

    title = f"Confusion matrix"

    normalize=None

    

    

    disp = plot_confusion_matrix(model, X_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize=normalize,

                                 values_format='.5g'

                                 )

    disp.ax_.set_title(title)

    plt.rcParams.update({'font.size': 16})

    #print(title)

    #print(disp.confusion_matrix)

    plt.grid(False)

    plt.show()    

    cm = confusion_matrix(predictions, y_test)

    tp = cm[0][0] 

    tn = cm[1][1]

    fp = cm[0][1]

    fn = cm[1][0]

    total = tp + tn + fp + fn

    print(f"tp for total: {tp}")

    print(f"tn: {tn}")

    print(f"fp: {fp}")

    print(f"fn: {fn}")

    accuracy = (tp + tn) / total

    precision = tp / (tp + fp)

    #***I think that True Positive Rate may be the indicator of a good

    #model....

    true_positive = tp / (tp + fn)

    print(f"The precision is: {precision}")

    print(f"The accuracy is {accuracy}")

    print(f"The prevalence of blue is {(tp + fn) / total}")

    print(f"The true_positive rate for total is {true_positive}")
run_model(total_train, total_train_y, df_test, y_test, LogisticRegression())
run_model(recent_train, recent_train_y, df_test, y_test, LogisticRegression())
run_model(total_train, total_train_y, df_test, y_test, RandomForestClassifier(random_state=0, min_samples_leaf=2))
run_model(recent_train, recent_train_y, df_test, y_test, RandomForestClassifier(random_state=0, min_samples_leaf=2))