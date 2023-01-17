import pandas as pd
import numpy as np
from IPython.core.display import Image

# We can display plots in the notebook using this line of code
%matplotlib inline
initial_evaluation = pd.read_csv('../input/task_1_data_meetup.csv')
initial_evaluation.head()
def get_TP_TN_FP_FN(df):
    
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    for i, row in df.iterrows():
        expected = row['expected']
        predicted = row['predicted']
        
        # True_postitives
        if predicted == None and expected == None:
            true_positives+=1

        # True_negatives
        elif predicted == None and expected == None:
            true_negatives+=1

        # False_positives
        elif predicted == None and expected == None:
            false_positives+=1
        
        # False_negatives
        elif predicted == None and expected == None:
            false_negatives+=1
        
    return true_positives, true_negatives, false_positives, false_negatives
true_positives, true_negatives, false_positives, false_negatives = get_TP_TN_FP_FN(initial_evaluation)
print('Number of true positives - {}'.format(true_positives))
print('Number of true negatives - {}'.format(true_negatives))
print('Number of false positives - {}'.format(false_positives))
print('Number of false negatives - {}'.format(false_negatives))
confusion_matrix = pd.crosstab(initial_evaluation['predicted'], initial_evaluation['expected'],
                               colnames=['Predicted'], rownames=['Expected'], margins=True)
confusion_matrix
def accuracy(TP, TN, FP, FN):
    return
accuracy_results = accuracy(true_positives, true_negatives, false_positives, false_negatives)
print('Model accuracy- {}'.format(accuracy_results))
def recall(TP, FN):
    return
recall_results = recall(true_positives, false_negatives)
print('Model recall- {}'.format(recall_results))
def precision(TP, FP):
    return
precision_results = precision(true_positives, false_positives)
print('Model precision- {}'.format(precision_results))
def F1(precision_result, recall_result):
    return None
f1_results = F1(precision_results, recall_results)
print('Model F1- {}'.format(f1_results))
print('Summary')
print('Accuracy- {}'.format(accuracy_results))
print('Recall- {}'.format(recall_results))
print('Precision- {}'.format(precision_results))
print('F1 score- {}'.format(f1_results))
new_model_data = pd.read_csv('../input/recall_data_meetup.csv')
true_positives, true_negatives, false_positives, false_negatives = get_TP_TN_FP_FN(new_model_data)

new_recall_results = recall(true_positives, false_negatives)
print('New model recall - {}'.format(new_recall_results))
new_precision_results = precision(None, None)
print('New model precision- {}'.format(new_precision_results))
new_accuracy_results = accuracy(None, None, None, None)
print('New model accuracy- {}'.format(new_accuracy_results))

new_F1_results = F1(new_precision_results, new_recall_results)
print('New model F1- {}'.format(new_F1_results))
print('Old Summary')
print('Accuracy- {}'.format(accuracy_results))
print('Recall- {}'.format(recall_results))
print('Precision- {}'.format(precision_results))
print('F1 score- {}'.format(f1_results))
print('New_Summary')
print('Accuracy- {}'.format(new_accuracy_results))
print('Recall- {}'.format(new_recall_results))
print('Precision- {}'.format(new_precision_results))
print('F1 score- {}'.format(new_F1_results))
new_model_data.apply(pd.Series.value_counts).plot(kind='bar')
final_data = pd.read_csv('../input/optimum_data_meetup.csv')
final_data.head()
expected_values = final_data['expected']
predicted_values = final_data['predicted']
from sklearn.metrics import accuracy_score
# final_idea_accuracy = accuracy_score(None, None)
# print('Final model accuracy- {}'.format(final_idea_accuracy))
from sklearn.metrics import recall_score
# final_idea_recall = recall_score(None, None)
# print('Final model recall- {}'.format(final_idea_recall))
from sklearn.metrics import precision_score
# final_idea_precision = precision_score(?, ?)
# print('Final model precision- {}'.format(final_idea_precision))
from sklearn.metrics import f1_score
# final_idea_f1 = f1_score(?, ?)
# print('Final model precision- {}'.format(final_idea_f1))
from sklearn.metrics import classification_report
print(classification_report(expected_values, predicted_values))
reg_data = pd.read_csv('../input/regression_data_task-1.csv')
reg_data.head()
import numpy as np
from matplotlib import pyplot as plt
X = reg_data['X']
y = reg_data['expected']
fig1 = plt.figure(1)
fig1.set_size_inches(10.5, 10.5)
frame1= fig1.add_axes((.1,.3,.8,.6))
plt.title('Predictions and Expected Values')
plt.plot(X, y,'.b') 
predictions = reg_data['predicted']
plt.plot(X, predictions,'-r')
plt.grid()

#Calculate difference
difference = predictions - y

frame2=fig1.add_axes((.1,.1,.8,.2))
plt.title('Residual Error')
plt.plot(X, difference, 'ob')
plt.plot(X, [0]*len(predictions), 'r')
plt.grid()
reg_predictions = reg_data['predicted']
reg_expected = reg_data['expected']
def MAE(prediction_values, expected_values):
    absolute_difference = abs(prediction_values-expected_values)
    return np.mean(absolute_difference)
mean_absolute_error_result = MAE(reg_predictions, reg_expected)
print('Model Mean Absolute Error- {}'.format(mean_absolute_error_result))
def RMSE(prediction_values, expected_values):
    difference = None
    squared_difference = None
    squared_difference_mean = None
    square_root = None  # Try np.sqrt
    return square_root
root_mean_square_error_result = RMSE(None, None)
print('Model Root Mean Square Error- {}'.format(root_mean_square_error_result))
reg_expected.plot.density()
reg_data_2 = pd.read_csv('../input/regression_data_task-2.csv')
new_reg_predicted = reg_data_2['predicted']
new_reg_expected = reg_data_2['expected']
# new_mean_absolute_error_result = MAE(None, None)
# print('Model Mean Absolute Error- {}'.format(new_mean_absolute_error_result))
new_root_mean_square_error_result = RMSE(None, None)
print('Model Root Mean Square Error- {}'.format(new_root_mean_square_error_result))
X_new = reg_data_2['X']
y_new = reg_data_2['expected']

fig2 = plt.figure(2)
fig2.set_size_inches(10.5, 10.5)
frame1= fig2.add_axes((.1,.3,.8,.6))
plt.title('Predictions and Expected Values')
plt.plot(X_new, new_reg_expected,'.b') 

# predictions_2 = np.dot(new_reg_predicted, coef_2)

plt.plot(X_new, new_reg_predicted,'-r')
plt.grid()

#Calculate difference
new_difference = new_reg_predicted - y_new

frame2= fig2.add_axes((.1,.1,.8,.2))
plt.title('Residual Error')
plt.plot(X_new, new_difference, 'ob')
plt.plot(X_new, [0]*len(X_new), 'r')
plt.ylim(-400, 300)
plt.grid()
new_difference.plot.hist()
new_difference.plot.density()
# add column for difference to dataframe
reg_data_2['difference'] = new_difference

def remove_outliers(df):
    # Calculate mean of differences
    mean = np.mean(df['difference'])
    
    # Calculate standard deviations of differences
    sd = np.std(df['difference'])
    
    # Filtering out values 2 standard deviations away from both sides of the mean.
    upper_removed = df[df['difference'] > mean - 2 * sd]
    df = upper_removed[upper_removed['difference'] < mean + 2 * sd]
    return df

outliers_removed_data = remove_outliers(reg_data_2)
X_clean = outliers_removed_data['X']
y_clean = outliers_removed_data['expected']
predictions_clean = outliers_removed_data['predicted']

fig3 = plt.figure(2)
fig3.set_size_inches(10.5, 10.5)
frame1= fig3.add_axes((.1,.3,.8,.6))
plt.title('Predictions and Expected Values')
plt.plot(X_clean, y_clean,'.b') 

plt.plot(X_clean, predictions_clean,'-r')
plt.grid()

#Calculate difference
difference_clean = predictions_clean - y_clean

frame2= fig3.add_axes((.1,.1,.8,.2))
plt.title('Residual Error')
plt.plot(X_clean, difference_clean, 'ob')
plt.plot(X_clean, [0]*len(X_clean), 'r')
plt.ylim(-400, 300)
plt.grid()
outliers_removed_data.head()
def remove_0_outliers(df):
    return
# cleaned_data = remove_0_outliers(outliers_removed_data)
# cleaned_data[cleaned_data['expected']==0]
### Expected output
# cleaned_data[cleaned_data['expected']==0]
# Uncomment this to run
# X_clean = cleaned_data['X']
# y_clean = cleaned_data['expected']
# predictions_clean = cleaned_data['predicted']

# fig3 = plt.figure(2)
# plt.title('ok')
# fig3.set_size_inches(10.5, 10.5)
# frame1= fig3.add_axes((.1,.3,.8,.6))
# plt.title('Predictions and Expected Values')
# plt.plot(X_clean, y_clean,'.b') 

# plt.plot(X_clean, predictions_clean,'-r')
# plt.grid()

# #Calculate difference
# difference_clean = predictions_clean - y_clean

# frame2= fig3.add_axes((.1,.1,.8,.2))
# plt.title('Residual Error')
# plt.plot(X_clean, difference_clean, 'ob')
# plt.plot(X_clean, [0]*185, 'r')
# plt.ylim(-400, 300)
# plt.grid()
# clean_predicted = cleaned_data['predicted']
# clean_expected = cleaned_data['expected']
from sklearn.metrics import mean_absolute_error
# cleaned_mean_absolute_error_result = mean_absolute_error(?, ?)
# print('Model Mean Absolute Error- {}'.format(cleaned_mean_absolute_error_result))
from sklearn.metrics import mean_squared_error

# cleaned_mean_square_error_result = mean_squared_error(None, None)
# cleaned_root_mean_square_error_result = None
# print('Model Root Mean Square Error- {}'.format(cleaned_root_mean_square_error_result))
banknotes_data = pd.read_csv('../input/banknotes_X.csv')
banknotes_labels = pd.read_csv('../input/banknotes_y.csv')
from sklearn.model_selection import train_test_split
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(banknotes_data, banknotes_labels, test_size=0.33, random_state=420)
from sklearn.linear_model import LogisticRegression
### Code here
from sklearn.datasets import load_diabetes
data_diabetes = load_diabetes()
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data_diabetes['data'], data_diabetes['target'], test_size=0.33, random_state=420)
from sklearn.linear_model import LinearRegression
### code here
