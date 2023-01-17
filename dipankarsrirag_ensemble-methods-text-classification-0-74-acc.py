#------------------------------------------Libraries---------------------------------------------------------------#
####################################################################################################################
#-------------------------------------Boiler Plate Imports---------------------------------------------------------#
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#---------------------------------------Text Processing------------------------------------------------------------#
import regex
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import WordPunctTokenizer
from string import punctuation
from nltk.stem import WordNetLemmatizer
#------------------------------------Metrics and Validation---------------------------------------------------------#
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
#-------------------------------------Models to be trained----------------------------------------------------------#
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost
#####################################################################################################################
train = pd.read_csv('/kaggle/input/ireland-historical-news/ireland-news-headlines.csv')
test = pd.read_csv('/kaggle/input/ireland-historical-news/w3-latnigrin-tokens.csv')
train.head()
train.info()
train.isna().sum()
train.headline_category.value_counts()
year = [] 
month = [] 
day = [] 

dates = train.publish_date.values

for date in dates:
    str_date = list(str(date))
    year.append(int("".join(str_date[0:4]))) 
    month.append(int("".join(str_date[4:6])))
    day.append(int("".join(str_date[6:8])))
train['year'] = year
train['month'] = month
train['day'] = day

train.drop(['publish_date'] , axis=1,inplace=True) 
train = train[train['headline_category'] != 'removed']
train.head()
print('Unique Headlines Categories: {}'.format(len(train.headline_category.unique())))
set([category for category in train.headline_category if "." not in category] ) 
train['headline_category'] = train.headline_category.apply(lambda x: x.split(".")[0]) 
train = train.loc[train.headline_category != 'removed']
wordnet_lemmatizer = WordNetLemmatizer()

stop = stopwords.words('english')

for punct in punctuation:
    stop.append(punct)

def filter_text(text, stop_words):
    word_tokens = WordPunctTokenizer().tokenize(text.lower())
    filtered_text = [regex.sub(u'\p{^Latin}', u'', w) for w in word_tokens if w.isalpha()]
    filtered_text = [wordnet_lemmatizer.lemmatize(w, pos="v") for w in filtered_text if not w in stop_words] 
    return " ".join(filtered_text)
train["filtered_text"] = train.headline_text.apply(lambda x : filter_text(x, stop)) 
train.head()
plt.figure(figsize=(10,5))
ax = sns.countplot(train.headline_category, palette = sns.color_palette("mako"))
plt.figure(figsize=(10,5))
ax = sns.lineplot(x=train.year.value_counts().index.values,y=train.year.value_counts().values, color = 'seagreen')
ax = plt.title('Number of Published News by Year')
plt.figure(figsize=(10,5))
ax = sns.lineplot(x=train.month.value_counts().index.values,y=train.month.value_counts().values, color = 'seagreen')
ax = plt.title('Number of Published News by Month')
plt.figure(figsize=(10,5))
ax = sns.lineplot(x=train.day.value_counts().index.values,y=train.day.value_counts().values, color = 'seagreen')
ax = plt.title('Number of Published News by Day')
def make_wordcloud(words,title):
    cloud = WordCloud(width=1920, height=1080,max_font_size=200, max_words=300, background_color="white").generate(words)
    plt.figure(figsize=(20,20))
    plt.imshow(cloud, interpolation="gaussian")
    plt.axis("off") 
    plt.title(title, fontsize=60)
    plt.show()
all_text = " ".join(train[train.headline_category == "news"].filtered_text) 
make_wordcloud(all_text, "News")
all_text = " ".join(train[train.headline_category == "culture"].filtered_text) 
make_wordcloud(all_text, "Culture")
all_text = " ".join(train[train.headline_category == "opinion"].filtered_text) 
make_wordcloud(all_text, "Opinion")
all_text = " ".join(train[train.headline_category == "business"].filtered_text) 
make_wordcloud(all_text, "Business")
all_text = " ".join(train[train.headline_category == "sport"].filtered_text) 
make_wordcloud(all_text, "Sport")
all_text = " ".join(train[train.headline_category == "lifestyle"].filtered_text) 
make_wordcloud(all_text, "Lifestyle")
tfidf = TfidfVectorizer(lowercase=False)
train_vec = tfidf.fit_transform(train['filtered_text'])
train_vec.shape
train['classification'] = train['headline_category'].replace(['news','culture','opinion','business','sport','lifestyle'],[0,1,2,3,4,5])
x_train, x_val, y_train, y_val = train_test_split(train_vec,train['classification'], stratify=train['classification'], test_size=0.2)
#C = np.arange(0, 1, 0.001)
#l1_ratio = np.ratio(0, 1, 0.01)
#max_iter = range(100, 500)
#warm_start = [True, False]
#solver = ['lbfgs', 'newton-cg']
#penalty = ['l2', 'l1']

#params = {
#    'C' : C,
#    'l1_ratio' : l1_ratio,
#    'max_iter' : max_iter,
#    'warm_start' : warm_start,
#    'solver' : solver,
#    'penalty' : penalty
#}
#
#random_search = RandomizedSearchCV(
#    estimator = LogisticRegression(),
#    param_distributions = params,
#    n_iter = 100,
#    cv = 3,
#    n_jobs = -1,
#    random_state = 1
#).fit(x_train, y_train)
#
#random_search.best_params_
model_lr = LogisticRegression(
    C=0.98, 
    l1_ratio=0.23, 
    max_iter=430, 
    random_state=1,
    warm_start=True
).fit(x_train, y_train)

model_lr.score(x_train, y_train)
predicted = model_lr.predict(x_val)

lr_acc = accuracy_score(y_val,predicted)
lr_cop = cohen_kappa_score(y_val,predicted)
lr = pd.DataFrame([lr_acc, lr_cop], columns = ['Logistic Regression with RandomizedSearchCV'])

print("Test score: {:.2f}".format(lr_acc))
print("Cohen Kappa score: {:.2f}".format(lr_cop))

plt.figure(figsize=(15,10))
ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)
ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',
            xticklabels=(['news','culture','opinion','business','sport','lifestyle']),
            yticklabels=(['news','culture','opinion','business','sport','lifestyle']))
#alpha = np.arange(0, 1, 0.001)
#fit_prior = [True, False]

#params = {
#    'alpha' : alpha,
#    'fit_prior' : fit_prior
#}
#
#random_search = RandomizedSearchCV(
#    estimator = MultinomialNB(),
#    param_distributions = params,
#    n_iter = 100,
#    cv = 3,
#    n_jobs = -1,
#    random_state = 1
#).fit(x_train, y_train)
#
#random_search.best_params_
model_mnb = MultinomialNB(alpha=1.9000000000000001, fit_prior=False).fit(x_train, y_train)

model_mnb.score(x_train, y_train)
predicted = model_mnb.predict(x_val)

mnb_acc = accuracy_score(y_val,predicted)
mnb_cop = cohen_kappa_score(y_val,predicted)
mnb = pd.DataFrame([mnb_acc, mnb_cop], columns = ['MultinomialNB with RandomizedSearchCV'])

print("Test score: {:.2f}".format(mnb_acc))
print("Cohen Kappa score: {:.2f}".format(mnb_cop))

plt.figure(figsize=(15,10))
ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)
ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',
            xticklabels=(['news','culture','opinion','business','sport','lifestyle']),
            yticklabels=(['news','culture','opinion','business','sport','lifestyle']))
model_sgd_hinge = SGDClassifier(
    loss='squared_hinge',
    penalty='l2',
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    n_jobs=-1,
    random_state=1,
    learning_rate='optimal',
    eta0=0.0,
    power_t=0.5,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    warm_start=False,
    average=False).fit(x_train, y_train)

model_sgd_hinge.score(x_train, y_train)
predicted = model_sgd_hinge.predict(x_val)

sgd_hinge_acc = accuracy_score(y_val,predicted)
sgd_hinge_cop = cohen_kappa_score(y_val,predicted)
sgd_hinge = pd.DataFrame([sgd_hinge_acc, sgd_hinge_cop], columns = ['SGDClassifier with Squared Hinge Loss'])

print("Test score: {:.2f}".format(sgd_hinge_acc))
print("Cohen Kappa score: {:.2f}".format(sgd_hinge_cop))
plt.figure(figsize=(15,10))
ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)
ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',
            xticklabels=(['news','culture','opinion','business','sport','lifestyle']),
            yticklabels=(['news','culture','opinion','business','sport','lifestyle']))
estimators = [
    ('svm', model_sgd_hinge),
    ('mnb', model_mnb),
    ('lr', model_lr)
]

estimators
model_voting = VotingClassifier(
    estimators = estimators,
    voting='hard', 
    n_jobs=-1,
    flatten_transform=True, 
    verbose=1).fit(x_train, y_train)

model_voting.score(x_train, y_train)
predicted = model_voting.predict(x_val)

voting_acc = accuracy_score(y_val,predicted)
voting_cop = cohen_kappa_score(y_val,predicted)
voting = pd.DataFrame([voting_acc, voting_cop], columns = ['Hard Voting Classifier'])

print("Test score: {:.2f}".format(voting_acc))
print("Cohen Kappa score: {:.2f}".format(voting_cop))

plt.figure(figsize=(15,10))
ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)
ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',
            xticklabels=(['news','culture','opinion','business','sport','lifestyle']),
            yticklabels=(['news','culture','opinion','business','sport','lifestyle']))
xgc = xgboost.XGBClassifier()

model_stack = StackingClassifier(
    estimators=estimators,
    final_estimator=xgc,
    n_jobs = -1,
    verbose = 1
)

model_stack.fit(x_train, y_train)

model_stack.score(x_train, y_train)
predicted = model_stack.predict(x_val)

stack_acc = accuracy_score(y_val,predicted)
stack_cop = cohen_kappa_score(y_val,predicted)
stack = pd.DataFrame([stack_acc, stack_cop], columns = ['Stacking Classifier'])

print("Test score: {:.2f}".format(stack_acc))
print("Cohen Kappa score: {:.2f}".format(stack_cop))

plt.figure(figsize=(15,10))
ax = sns.heatmap(confusion_matrix(y_val,predicted),annot=True)
ax = ax.set(xlabel='Predicted',ylabel='True',title='Confusion Matrix',
            xticklabels=(['news','culture','opinion','business','sport','lifestyle']),
            yticklabels=(['news','culture','opinion','business','sport','lifestyle']))
model_comp = pd.concat([lr, mnb, sgd_hinge, voting, stack], axis = 1)
model_comp
test_vec = tfidf.transform(test.iloc[:, -1])
test_vec.shape
cat = ['news','culture','opinion','business','sport','lifestyle']
code = [0,1,2,3,4,5]
dic = dict([(code[x], cat[x])for x in range(6)])
dic
pred = model_stack.predict(test_vec)

predictions = []
for i in pred:
    predictions.append(dic[i])
test['Category'] = predictions

test.head()
plt.figure(figsize=(10,5))
ax = sns.countplot(test.Category, palette = sns.color_palette("mako"))