# imported necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
#from sklearn.cross_validation import cross_val_score
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn import model_selection
#from sklearn import cross_validation
from scipy.stats import uniform
from sklearn.model_selection import TimeSeriesSplit
warnings.filterwarnings("ignore")
import sqlite3
con = sqlite3.connect('finalassignment.sqlite')
cleaned_data = pd.read_sql_query('select * from Reviews', con)
cleaned_data.shape
# Sort data based on time
cleaned_data["Time"] = pd.to_datetime(cleaned_data["Time"], unit = "s")
cleaned_data = cleaned_data.sort_values(by = "Time")
cleaned_data.shape
cleaned_data['Score'].value_counts()
# To randomly sample 5k points from both class

data_p = cleaned_data[cleaned_data['Score'] == 'positive'].sample(n = 5000)
data_n = cleaned_data[cleaned_data['Score'] == 'negative'].sample(n = 5000)
final_10k = pd.concat([data_p, data_n])
final_10k.shape
# converting scores in 0 and 1
final_10k["Score"] = final_10k["Score"].map(lambda x: 1 if x == "positive" else 0)
#encoded_labels = df['label'].map(lambda x: 1 if x == 'spam' else 0).values
# Sorting data based on time
final_10k['Time'] = pd.to_datetime(final_10k['Time'], unit = 's')
final_10k = final_10k.sort_values(by = 'Time')
final_10k.shape
# Grid search
def lr_grid_plot(X_train, y_train):
    tuned_parameters_grid = [{'penalty': ['l1','l2'],'C': [10**-4, 10**-2, 10**0, 10**2, 10**4]}]
    cv = TimeSeriesSplit(n_splits = 3)
    model_lr_grid = GridSearchCV(LogisticRegression(), param_grid = tuned_parameters_grid, cv = cv)
    model_lr_grid.fit(X_train, y_train)
    print("\n**********GridSearchCV**********\n")
    print("\nOptimal C:", model_lr_grid.best_estimator_.C)
    print('\nBest penalty:', model_lr_grid.best_estimator_.get_params()['penalty'])
    score = model_lr_grid.cv_results_
    plot_df = pd.DataFrame(score)
    plt.plot(plot_df["param_C"], 1- plot_df["mean_test_score"], "-o")
    plt.title("CV Error vs C")
    plt.xlabel("C")
    plt.ylabel("Cross-validation Error")
    plt.show()
    return model_lr_grid.best_estimator_.C
# 10k data which will use to train model after vectorization
X = final_10k["CleanedText"]
print("shape of X:", X.shape)
# class label
y = final_10k["Score"]
print("shape of y:", y.shape)
# split data into train and test where 70% data used to train model and 30% for test
from sklearn.model_selection import train_test_split
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, shuffle = False)
print(X_train.shape, y_train.shape, x_test.shape , y_test.shape)
# Train Vectorizor
from sklearn.feature_extraction.text import CountVectorizer 
bow = CountVectorizer()
X_train = bow.fit_transform(X_train)
X_train
# Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean = False)
std_X_train = scaler.fit_transform(X_train)
# Test Vectorizor
x_test = bow.transform(x_test)
x_test.shape
scaler = StandardScaler(with_mean = False)
std_x_test = scaler.fit_transform(x_test)
std_x_test.shape
# To choose optimal c using cross validation
from sklearn.model_selection import TimeSeriesSplit
optimal_lambda_bow_grid = lr_grid_plot(std_X_train, y_train)
optimal_lambda_bow_grid
# instantiate learning model 
lr_model =  LogisticRegression(penalty = 'l2', C = optimal_lambda_bow_grid, class_weight = "balanced")
# fitting the model
lr_model.fit(std_X_train, y_train)
# predict the response
pred = lr_model.predict(std_x_test)
# predict probablistic response
pred_prob = lr_model.predict_proba(std_x_test)

# this step use both technique
train_acc_bow_grid = lr_model.score(std_X_train, y_train)
print("Train accuracy:",train_acc_bow_grid)
test_acc_bow_grid = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the logistic regression for c = %f is %.2f%%' % (optimal_lambda_bow_grid, test_acc_bow_grid))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
models = pd.DataFrame({'Model': ['Logistic Regression with Bow'], 'Hyper Parameter(K) for grid search': [optimal_lambda_bow_grid], 'Train Error': [train_acc_bow_grid], 'Test Error': [100 - test_acc_bow_grid], 'Accuracy': [test_acc_bow_grid ], 'Train Accuracy': [train_acc_bow_grid ]}, columns = ["Model", "Hyper Parameter(K) for grid search", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
models.sort_values(by='Accuracy', ascending=False)
def plot_precision_recall_curve(recall, precision):
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision_recall_curve")
    plt.plot(recall, precision, "-o")
    plt.show()
# F1-score, auc, precision_recall_curve
from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
f1 = f1_score(y_test, pred)
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])
auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, pred_prob[:,1])
print("Average precision score:", avg_precision)
print("F1_score:", f1)
print("Auc score:",auc)
plot_precision_recall_curve(recall, precision)
# Random search
from scipy.stats import uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def lr_random_plot(X_train, y_train):
    tuned_parameters_random = {'penalty': ['l1','l2'], 'C': uniform(loc = 0, scale = 4)}
    cv = TimeSeriesSplit(n_splits = 3)
    model_lr_random = RandomizedSearchCV(LogisticRegression(), tuned_parameters_random, cv = cv, n_iter = 10)
    model_lr_random.fit(X_train, y_train)
    print("\n\n**********RandomizedSearchCV**********\n")
    print("\nOptimal C:", model_lr_random.best_estimator_.C)
    print('\nBest penalty:', model_lr_random.best_estimator_.get_params()['penalty'])
    score = model_lr_random.cv_results_
    plot_df = pd.DataFrame(score)
    plt.plot(plot_df["param_C"], 1 - plot_df["mean_test_score"], "-o")
    plt.title("CV Error vs C")
    plt.xlabel("C")
    plt.ylabel("Cross-validation Error")
    plt.show()
    return model_lr_random.best_estimator_.C
optimal_lambda_bow_random = lr_random_plot(std_X_train, y_train)
optimal_lambda_bow_random
# instantiate learning model 
lr_model =  LogisticRegression(penalty = 'l2', C = optimal_lambda_bow_grid, class_weight = "balanced")
# fitting the model
lr_model.fit(std_X_train, y_train)
# predict the response
pred = lr_model.predict(std_x_test)
# predict probablistic response
pred_prob = lr_model.predict_proba(std_x_test)

# this step use both technique
train_acc_bow_random = lr_model.score(std_X_train, y_train)
print("Train accuracy:",train_acc_bow_random)
test_acc_bow_random = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the logistic regression for c = %f is %.2f%%' % (optimal_lambda_bow_random, test_acc_bow_random))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
modelsgrid = pd.DataFrame({'Model': ['Logistic Regression with Bow'], 'Hyper Parameter(K) for random search': [optimal_lambda_bow_random], 'Train Error': [train_acc_bow_random], 'Test Error': [100 - test_acc_bow_random], 'Accuracy': [test_acc_bow_random ], 'Train Accuracy': [train_acc_bow_random ]}, columns = ["Model", "Hyper Parameter(K) for random search", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
modelsgrid.sort_values(by='Accuracy', ascending=False)

def plot_precision_recall_curve(recall, precision):
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision_recall_curve")
    plt.plot(recall, precision, "-o")
    plt.show()
# F1-score, auc, precision_recall_curve
from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
f1 = f1_score(y_test, pred)
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])
auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, pred_prob[:,1])
print("Average precision score:", avg_precision)
print("F1_score:", f1)
print("Auc score:",auc)
plot_precision_recall_curve(recall, precision)
# Tried different value of c and finding features weight
# More Sparsity (Fewer elements of W* being non-zero) by increasing Lambda (decreasing C)
C_param = [10, 1, 0.1]

for c in C_param:
    clf = LogisticRegression(penalty='l1', C = c, class_weight = "balanced",  solver='liblinear')
    clf.fit(X_train, y_train)
    print('\nC value:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy: %0.2f%%' %(clf.score(std_X_train, y_train) * 100))
    print('Test accuracy: %0.2f%%' %(clf.score(std_x_test, y_test) * 100))
    print("Number of non-zero element: ",np.count_nonzero(clf.coef_))
clf = LogisticRegression(penalty='l1', C = optimal_lambda_bow_random, class_weight = "balanced" ,solver='liblinear')
clf.fit(std_X_train, y_train)
y_pred = clf.predict(std_x_test)
print("Accuracy score: %0.2f%%" %(accuracy_score(y_test, y_pred) * 100))
print(np.count_nonzero(clf.coef_))
std_X_train.shape
from scipy.sparse import find

# Before adding noise in data
cf = clf.coef_[0]
w_coef1 = cf[np.nonzero(cf)]
print(w_coef1[:20])
# Generate random normal variable as a noise 
std_X_train_pert = std_X_train
noise = np.random.normal(0, 0.0001, size = (std_X_train_pert[np.nonzero(std_X_train_pert)].size))
#print(noise.shape)
np.nonzero(std_X_train_pert)
std_X_train_pert[np.nonzero(std_X_train_pert)] = noise + std_X_train_pert[np.nonzero(std_X_train_pert)]
std_X_train_pert.shape
clf = LogisticRegression(penalty ='l1', C = optimal_lambda_bow_random, class_weight = "balanced", solver='liblinear')
clf.fit(std_X_train_pert, y_train)
y_pred = clf.predict(std_x_test)
print("Accuracy score: %0.2f%%" %(accuracy_score(y_test, y_pred) * 100))
print(np.count_nonzero(clf.coef_))
cf = clf.coef_[0]
w_coef2 = cf[np.nonzero(cf)]
print(w_coef2[:20])
# Calculate %increase 
cnt = 0
for w1, w2 in zip(w_coef1, w_coef2):
    inc = abs(w1 - w2)/abs(w1) * 100
    if inc > 40:
        cnt += 1
print("No of weights that changes more than 40% is:", cnt)
# Features importance 

features = bow.get_feature_names()
coef = clf.coef_[0]
coeff_df = pd.DataFrame({'Word' : features, 'Coefficient' : coef})
coeff_df = coeff_df.sort_values("Coefficient", ascending = False)
print('*----------****Top 10 positive*------------------****')
print(coeff_df.head(10))
print('**-------------***Top 10 negative**---------------***')
print(coeff_df.tail(10))
# data
X = final_10k["CleanedText"]
# Target/class-label
y = final_10k["Score"]
# Split data
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, shuffle = False)
print(X_train.shape, x_test.shape, y_train.shape, y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
X_train = tf_idf_vect.fit_transform(X_train)
X_trn = X_train
X_train
# Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean = False)
std_X_train = scaler.fit_transform(X_train)
# Convert test text data to its vectorizor
x_test = tf_idf_vect.transform(x_test)
x_tst = x_test
x_test.shape
scaler = StandardScaler(with_mean = False)
std_x_test = scaler.fit_transform(x_test)
# To choose optimal_alpha using nested cross validation
optimal_lambda_tfidf_grid = lr_grid_plot(std_X_train, y_train)
optimal_lambda_tfidf_grid
# instantiate learning model 
lr_model =  LogisticRegression(penalty = 'l2', C = optimal_lambda_tfidf_grid, class_weight = "balanced")
# fitting the model
lr_model.fit(std_X_train, y_train)
# predict the response
pred = lr_model.predict(std_x_test)
# predict probablistic response
pred_prob = lr_model.predict_proba(std_x_test)
# F1-score, auc, precision_recall_curve
from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
f1 = f1_score(y_test, pred)
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])
auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, pred_prob[:,1])
print("Average precision score:", avg_precision)
print("F1_score:", f1)
print("Auc score:",auc)
plot_precision_recall_curve(recall, precision)
train_acc_tfidf_grid = lr_model.score(std_X_train, y_train)
print("Train accuracy:",train_acc_tfidf_grid)
test_acc_tfidf_grid = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the logistic regression for c = %f is %.2f%%' % (optimal_lambda_tfidf_grid, test_acc_tfidf_grid))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
modelsgrid = pd.DataFrame({'Model': ['Logistic Regression with TFIDF'], 'Hyper Parameter(K) for grid search': [optimal_lambda_tfidf_grid], 'Train Error': [train_acc_tfidf_grid], 'Test Error': [100 - test_acc_tfidf_grid], 'Accuracy': [test_acc_tfidf_grid ], 'Train Accuracy': [train_acc_tfidf_grid ]}, columns = ["Model", "Hyper Parameter(K) for grid search", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
modelsgrid.sort_values(by='Accuracy', ascending=False)

optimal_lambda_tfidf_random = lr_random_plot(std_X_train, y_train)
optimal_lambda_tfidf_random
# instantiate learning model 
lr_model =  LogisticRegression(penalty = 'l2', C = optimal_lambda_tfidf_random, class_weight = "balanced")
# fitting the model
lr_model.fit(std_X_train, y_train)
# predict the response
pred = lr_model.predict(std_x_test)
# predict probablistic response
pred_prob = lr_model.predict_proba(std_x_test)
# F1-score, auc, precision_recall_curve
from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
f1 = f1_score(y_test, pred)
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])
auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, pred_prob[:,1])
print("Average precision score:", avg_precision)
print("F1_score:", f1)
print("Auc score:",auc)
plot_precision_recall_curve(recall, precision)
train_acc_tfidf_random = lr_model.score(std_X_train, y_train)
print("Train accuracy %f%%:" % (train_acc_tfidf_random))
test_acc_tfidf_random = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the logistic regression for c = %f is %.2f%%' % (optimal_lambda_tfidf_random, test_acc_tfidf_random))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
modelsgrid = pd.DataFrame({'Model': ['Logistic Regression with TFIDF'], 'Hyper Parameter(K) for random search': [optimal_lambda_tfidf_random], 'Train Error': [train_acc_tfidf_random], 'Test Error': [100 - test_acc_tfidf_random], 'Accuracy': [test_acc_tfidf_random ], 'Train Accuracy': [train_acc_tfidf_random ]}, columns = ["Model", "Hyper Parameter(K) for random search", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
modelsgrid.sort_values(by='Accuracy', ascending=False)

# Tried different value of c and finding features weight
# More Sparsity (Fewer elements of W* being non-zero) by increasing Lambda (decreasing C)
C_param = [10, 1, 0.1]

for c in C_param:
    clf = LogisticRegression(penalty = 'l2', C = c, class_weight = "balanced")
    clf.fit(X_train, y_train)
    print('\nC value:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy: %0.3f%%' %(clf.score(std_X_train, y_train) * 100))
    print('Test accuracy: %0.3f%%' %(clf.score(std_x_test, y_test) * 100))
    print("Number of non-zero element: ",np.count_nonzero(clf.coef_))
clf = LogisticRegression(penalty = 'l1', C = optimal_lambda_tfidf_grid, class_weight = "balanced" ,solver='liblinear')
clf.fit(std_X_train, y_train)
y_pred = clf.predict(std_x_test)
print("Accuracy score: %.2f%%" %(accuracy_score(y_test, y_pred) * 100))
print(np.count_nonzero(clf.coef_))
std_X_train.shape
np.count_nonzero(clf.coef_)
from scipy.sparse import find

# Before adding noise in data
cf = clf.coef_[0]
w_coef1 = cf[np.nonzero(cf)]
print(w_coef1[:20])
# Generate random normal variable as a noise 
std_X_train_pert = std_X_train
noise = np.random.normal(0, 0.001, size = (std_X_train_pert[np.nonzero(std_X_train_pert)].size,))
#print(noise.shape)
np.nonzero(std_X_train_pert)
std_X_train_pert[np.nonzero(std_X_train_pert)] = noise + std_X_train_pert[np.nonzero(std_X_train_pert)]
std_X_train_pert.shape
std_X_train_pert.shape
clf = LogisticRegression(penalty = 'l2', C = optimal_lambda_tfidf_grid, class_weight = "balanced")
clf.fit(std_X_train_pert, y_train)
y_pred = clf.predict(std_x_test)
print("Accuracy score: %0.2f%%" %(accuracy_score(y_test, y_pred) * 100))
print(np.count_nonzero(clf.coef_))
np.count_nonzero(clf.coef_)
cf = clf.coef_[0]
w_coef2 = cf[np.nonzero(cf)]
print(w_coef2[:20])
# Calculate %increase 
cnt = 0
for w1, w2 in zip(w_coef1, w_coef2):
    inc = abs(w1 - w2)/abs(w1) * 100
    if inc > 40:
        cnt += 1
print("No of weights that changes more than 40% is:", cnt)
# Features importance 

features = tf_idf_vect.get_feature_names()
coef = clf.coef_[0]
coeff_df = pd.DataFrame({'Word' : features, 'Coefficient' : coef})
coeff_df = coeff_df.sort_values('Coefficient', ascending = 0)
print('*****Top 10 positive*****')
print(coeff_df.head(10))
print('*****Top 10 negative*****')
print(coeff_df.tail(10))
# data
X = final_10k["Text"]
X.shape
# Target/class-label
y = final_10k["Score"]
y.shape
# Split data
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42, shuffle = False)
print(X_train.shape, x_test.shape, y_train.shape, y_test.shape)
import re
def cleanhtml(sentence): #function to clean the word of any html-tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext
def cleanpunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned
# Train your own Word2Vec model using your own train text corpus
import gensim
list_of_sent=[]
for sent in X_train:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent.append(filtered_sentence)
w2v_model_train = gensim.models.Word2Vec(list_of_sent, min_count = 5, size = 50, workers = 4)
w2v_model_train.wv.most_similar('like')
w2v_train = w2v_model_train[w2v_model_train.wv.vocab]
w2v_train.shape
# Train your own Word2Vec model using your own test text corpus
import gensim
list_of_sent_test = []
for sent in x_test:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent_test.append(filtered_sentence)
w2v_model_test = gensim.models.Word2Vec(list_of_sent_test, min_count = 5, size = 50, workers = 4)
w2v_model_test.wv.most_similar('like')
w2v_test = w2v_model_test[w2v_model_test.wv.vocab]
w2v_test.shape
# average Word2Vec
# compute average word2vec for each review.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sent: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model_train.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
print(len(sent_vectors))
print(len(sent_vectors[0]))
# average Word2Vec
# compute average word2vec for each review.
sent_vectors_test = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sent_test: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model_test.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors_test.append(sent_vec)
print(len(sent_vectors_test))
print(len(sent_vectors_test[0]))
X_train = sent_vectors
#X_train
# Standardization 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean = False)
std_X_train = scaler.fit_transform(X_train)
x_test = sent_vectors_test
#x_test
scaler = StandardScaler(with_mean = False)
std_x_test = scaler.fit_transform(x_test)
# To choose optimal_alpha using nested cross validation
#from sklearn.model_selection import KFold
#from sklearn.model_selection import KFold
optimal_lambda_avgw2v_grid = lr_grid_plot(std_X_train, y_train)
optimal_lambda_avgw2v_grid
# instantiate learning model 
lr_model =  LogisticRegression(penalty = 'l2', C = optimal_lambda_avgw2v_grid, class_weight = "balanced")
# fitting the model
lr_model.fit(std_X_train, y_train)
# predict the response
pred = lr_model.predict(std_x_test)
# predict probablistic response
pred_prob = lr_model.predict_proba(std_x_test)
# F1-score, auc, precision_recall_curve
from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
f1 = f1_score(y_test, pred)
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])
auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, pred_prob[:,1])
print("Average precision score:", avg_precision)
print("F1_score:", f1)
print("Auc score:",auc)
plot_precision_recall_curve(recall, precision)
train_acc_avgw2v_grid = lr_model.score(std_X_train, y_train)
print("Train accuracy:", train_acc_avgw2v_grid)
test_acc_avgw2v_grid = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the logistic regression for c = %f is %.2f%%' % (optimal_lambda_avgw2v_grid, test_acc_avgw2v_grid))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
modelsgrid = pd.DataFrame({'Model': ['Logistic Regression with AvgW2V'], 'Hyper Parameter(K) for random search': [optimal_lambda_avgw2v_grid], 'Train Error': [train_acc_avgw2v_grid], 'Test Error': [100 - test_acc_avgw2v_grid], 'Accuracy': [test_acc_avgw2v_grid ], 'Train Accuracy': [train_acc_avgw2v_grid ]}, columns = ["Model", "Hyper Parameter(K) for grid  search", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
modelsgrid.sort_values(by='Accuracy', ascending=False)

optimal_lambda_avgw2v_random = lr_random_plot(std_X_train, y_train)
optimal_lambda_avgw2v_random
# instantiate learning model 
lr_model =  LogisticRegression(penalty = 'l2', C = optimal_lambda_avgw2v_random, class_weight = "balanced")
# fitting the model
lr_model.fit(std_X_train, y_train)
# predict the response
pred = lr_model.predict(std_x_test)
# predict probablistic response
pred_prob = lr_model.predict_proba(std_x_test)
# F1-score, auc, precision_recall_curve
from sklearn.metrics import f1_score, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import auc
f1 = f1_score(y_test, pred)
precision, recall, thresholds = precision_recall_curve(y_test, pred_prob[:,1])
auc = auc(recall, precision)
avg_precision = average_precision_score(y_test, pred_prob[:,1])
print("Average precision score:", avg_precision)
print("F1_score:", f1)
print("Auc score:", auc)
plot_precision_recall_curve(recall, precision)
# Accuracy on train data
train_acc_avgw2v_random = lr_model.score(std_X_train, y_train)
print("Train accuracy", train_acc_avgw2v_random)
test_acc_avgw2v_random = accuracy_score(y_test, pred) * 100
print('\nThe accuracy of the logistic regression for c = %f is %.2f%%' % (optimal_lambda_avgw2v_random, test_acc_avgw2v_random))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)
cm
# plot confusion matrix to describe the performance of classifier.
import seaborn as sns
class_label = ["negative", "positive"]
df_cm = pd.DataFrame(cm, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusiion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
# To show main classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
# model for knn with bag of word
modelsgrid = pd.DataFrame({'Model': ['Logistic Regression with AvgW2V'], 'Hyper Parameter(K) for random search': [optimal_lambda_avgw2v_random], 'Train Error': [train_acc_avgw2v_random], 'Test Error': [100 - test_acc_avgw2v_random], 'Accuracy': [test_acc_avgw2v_random ], 'Train Accuracy': [train_acc_avgw2v_random ]}, columns = ["Model", "Hyper Parameter(K) for random  search", "Train Error", "Test Error", "Accuracy" , "Train Accuracy"])
modelsgrid.sort_values(by='Accuracy', ascending=False)

# Tried different value of c and finding features weight
# More Sparsity (Fewer elements of W* being non-zero) by increasing Lambda (decreasing C)
C_param = [10, 1, 0.1]

for c in C_param:
    clf = LogisticRegression(penalty = 'l2', C = c, class_weight = "balanced")
    clf.fit(std_X_train, y_train)
    print('\nC value:', c)
    print('Coefficient of each feature:', clf.coef_)
    print('Training accuracy: %0.3f%%' %(clf.score(std_X_train, y_train) * 100))
    print('Test accuracy: %0.3f%%' %(clf.score(std_x_test, y_test) * 100))
    print("Number of non-zero element: ",np.count_nonzero(clf.coef_))
clf = LogisticRegression(penalty = 'l2', C = optimal_lambda_avgw2v_random, class_weight = "balanced")
clf.fit(std_X_train, y_train)
y_pred = clf.predict(std_x_test)
print("Accuracy score: %0.2f%%" %(accuracy_score(y_test, y_pred) * 100))
print(np.count_nonzero(clf.coef_))
std_X_train.shape
np.count_nonzero(clf.coef_)
from scipy.sparse import find

# Before adding noise in data
cf = clf.coef_[0]
w_coef1 = cf[np.nonzero(cf)]
print(w_coef1[:50])
# Generate random normal variable as a noise 
std_X_train_pert = std_X_train
noise = np.random.normal(0, 0.001, size = (std_X_train_pert[np.nonzero(std_X_train_pert)].size,))
#print(noise.shape)
np.nonzero(std_X_train_pert)
std_X_train_pert[np.nonzero(std_X_train_pert)] = noise + std_X_train_pert[np.nonzero(std_X_train_pert)]
std_X_train_pert.shape
std_X_train_pert.shape
clf = LogisticRegression(penalty = 'l2', C = optimal_lambda_avgw2v_random, class_weight = "balanced")
clf.fit(std_X_train_pert, y_train)
y_pred = clf.predict(std_x_test)
print("Accuracy score: %0.2f%%" %(accuracy_score(y_test, y_pred) * 100))
print(np.count_nonzero(clf.coef_))
cf = clf.coef_[0]
w_coef2 = cf[np.nonzero(cf)]
print(w_coef2[:50])
# Calculate %increase 
cnt = 0
for w1, w2 in zip(w_coef1, w_coef2):
    inc = (abs(w1 - w2)/abs(w2)) * 100
    if inc > 40:
        cnt += 1
print("No of weights that changes more than 40% is:", cnt)
# model performence table using grid search
#import itables
models = pd.DataFrame({'Model': ['LogisticRegression with Bow', "LogisticRegression with TFIDF", "LogisticRegression with avgw2v"], 'Hyper Parameter(lambda)': [optimal_lambda_bow_grid, optimal_lambda_tfidf_grid, optimal_lambda_avgw2v_grid], 'Train Error': [1-train_acc_bow_grid, 1-train_acc_tfidf_grid, 1-train_acc_avgw2v_grid], 'Test Error': [100-test_acc_bow_grid, 100-test_acc_tfidf_grid, 100-test_acc_avgw2v_grid], 'Accuracy': [test_acc_bow_grid, test_acc_tfidf_grid, test_acc_avgw2v_grid]}, columns = ["Model", "Hyper Parameter(lambda)", "Train Error", "Test Error", "Accuracy"]).sort_values(by='Accuracy', ascending=False)
models.sort_values(by='Accuracy', ascending=False)
# model performence table using random search
models = pd.DataFrame({'Model': ['LogisticRegression with Bow', "LogisticRegression with TFIDF", "LogisticRegression with avgw2v"], 'Hyper Parameter(lambda)': [optimal_lambda_bow_random, optimal_lambda_tfidf_random, optimal_lambda_avgw2v_random], 'Train Error': [1-train_acc_bow_random, 1-train_acc_tfidf_random, 1-train_acc_avgw2v_random], 'Test Error': [100-test_acc_bow_random, 100-test_acc_tfidf_random, 100-test_acc_avgw2v_random], 'Accuracy': [test_acc_bow_random, test_acc_tfidf_random, test_acc_avgw2v_random]}, columns = ["Model", "Hyper Parameter(lambda)", "Train Error", "Test Error", "Accuracy"]).sort_values(by = "Accuracy", ascending = False)
models.sort_values(by='Accuracy', ascending=False)