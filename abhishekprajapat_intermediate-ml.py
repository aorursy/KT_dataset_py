from IPython.display import HTML
from IPython.lib.display import YouTubeVideo
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/pd-0G0MigUA?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
HTML('<center><iframe width="700" height="400" src="https://www.youtube.com/embed/YWFqtmGK0fk?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe></center>')
data = pd.read_csv("../input/titanic/train.csv")
data.head(2)
data.isna().sum()
data.drop(columns = 'PassengerId', inplace=True)
# I will only use numerical values to make work short but you should do it using all options available
filtered_data = data[['Pclass', 'Fare', 'SibSp', 'Parch', 'Age']]
train = filtered_data[filtered_data['Age'].notnull()]
test = filtered_data[filtered_data['Age'].isna()]

filtered_data.shape ,train.shape, test.shape
linear_regressor = LinearRegression()

linear_regressor.fit(train.drop(columns = 'Age'), train['Age'])

test['Age'] = linear_regressor.predict(test.drop(columns = 'Age'))
test.isna().sum()
data['Cabin'].notnull().sum()
data['Cabin'][data['Cabin'].notnull()][:5]
floor = []
for i in data['Cabin']:
    
    try:
        value = list(i)[0]
        floor.append(value)
    except:
        floor.append(i)
        
data['floor'] = floor
print(data['floor'][data['floor'].notnull()][:5])
print("Unique values in floor column: ", len(data['floor'].unique()))
print('Unique values in Cabin column: ', len(data['Cabin'].unique()))
data.head(2)
titles = []
for i in data['Name']:
    value = (i.split(',')[1]).split(' ')[1]
    titles.append(value)
data['titles'] = titles
px.bar(data, x = 'titles', color='titles')
for i in range(len(titles)):
    if titles[i] not in ['Mr.', 'Mrs.', 'Miss.', 'Master.']:
        titles[i] = 'other'

data['titles'] = titles
px.bar(data, x = 'titles', color='titles', facet_col='Survived')
data = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip')
data.sample(2)
# Which values you would have to predict
cols_target = ['obscene','insult','toxic','severe_toxic','identity_hate','threat']

values = list(data[cols_target].sum())
target = cols_target

other_rows = np.array(values).sum()

neutral_comment = data.shape[0] - other_rows

values.append(neutral_comment)
target.append('Total_rows')

ax = sns.barplot(target, values)
ax.set_xticklabels(target, rotation=45)
ax.set_title('Number of rows per type', fontsize=17)
ax.set_ylabel('Number of rows', fontsize=14)
ax.set_xlabel('Comment type', fontsize=14);
from sklearn.manifold import TSNE
import imblearn
from sklearn.model_selection import StratifiedKFold
k_fold = StratifiedKFold(n_splits=10)
for fold, (train, validation) in enumerate(k_fold.split(X=data, y=data.toxic.values)):
    data.loc[validation, 'kfold'] = fold
from skmultilearn.problem_transform import ClassifierChain, BinaryRelevance
clf_chain = ClassifierChain()
bin_rel = BinaryRelevance()
def chain_classifier(model, x_train, y_train, x_validation=None, y_validation=None, 
                     x_test=None, y_test=None, validate=False, test=False, display=False, tune=False, param=None):

    print('Model = ', model, '\n')
    train_scores = []
    validation_scores = []
    test_scores = []
    model_details = []
    
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import accuracy_score, f1_score
    
    for label in cols_target:

        print('... Processing {} \n'.format(label))
        
        y_train_label = y_train[label]
        y_validation_label = y_validation[label]
        y_test_label = y_test[label]
        
        # To tune the model
        if tune:
            model = RandomizedSearchCV(model, param, n_jobs=-1, cv=10)
        
        # train the model using x_train & y_train
        model.fit(x_train,y_train_label)
        
        # compute the training results
        y_train_pred = model.predict(x_train)
        
        if display:
            print('Training Accuracy is {}'.format(accuracy_score(y_train_label, y_train_pred)))
            print('Training F1Score is {} \n'.format(f1_score(y_train_label, y_train_pred)))
        
        # Append scores
        to_append = (accuracy_score(y_train_label, y_train_pred), f1_score(y_train_label, y_train_pred))
        train_scores.append(to_append)
        
        # Adding predictions as features
        x_train = add_feature(x_train, y_train_pred)
        
        if validate:
            # compute validation results
            y_validation_pred = model.predict(x_validation)
            
            if display:
                print('Validation Accuracy is {}'.format(accuracy_score(y_validation_label, y_validation_pred)))
                print('Validation F1Score is {} \n'.format(f1_score(y_validation_label, y_validation_pred)))
            
            # Adding prediction as feature
            x_validation = add_feature(x_validation, y_validation_pred)
            
            
            # Append scores
            to_append = (accuracy_score(y_validation_label, y_validation_pred), f1_score(y_validation_label, y_validation_pred))
            validation_scores.append(to_append)

        if test:
            # compute test results
            y_test_pred = model.predict(x_test)
            
            if display:
                print('Test Accuracy is {}'.format(accuracy_score(y_test_label, y_test_pred)))
                print('Test F1Score is {} \n'.format(f1_score(y_test_label, y_test_pred)))
            
            # append Scores
            to_append = (accuracy_score(y_test_label, y_test_pred), f1_score(y_test_label, y_test_pred))
            test_scores.append(to_append)
            
            # Adding prediction as feature
            x_test = add_feature(x_test, y_test_pred)
        
        model_details.append(model)
        
        
    scores = (train_scores, validation_scores, test_scores)    
        
    return scores, model_details