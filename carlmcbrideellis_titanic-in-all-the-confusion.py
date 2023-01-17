import pandas  as pd

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score



# read in the files that I am going to use:

gender   = pd.read_csv('../input/titanic/gender_submission.csv')  

perfect  = pd.read_csv('../input/submission-solution/submission_solution.csv')

allZeros = pd.read_csv('../input/titanic-all-zeros-csv-file/all_0s.csv')

accuracy_score( perfect['Survived'] , gender['Survived'] )
titanic_cm = confusion_matrix( perfect['Survived'] , gender['Survived'] )

print(titanic_cm)
# convert the ndarray to a pandas dataframe

cm_df = pd.DataFrame(titanic_cm)

# set the size of the figure

plt.figure(figsize = (5,5))

sn.heatmap(cm_df, 

           annot=True, annot_kws={"size": 25},

           fmt="d",         # decimals format

           xticklabels=False, 

           yticklabels=False,

           cmap="viridis", 

           cbar=False)

plt.show()
tn, fp, fn, tp = confusion_matrix(perfect['Survived'] , gender['Survived']).ravel()

print("Number of true negatives  (tn) = ",tn)

print("Number of true positives  (tp) = ",tp)

print("Number of false negatives (fn) = ",fn)

print("Number of false positives (fp) = ",fp)

print("Precision                                          = tp / (tp + fp) =", tp / (tp + fp))

print("Recall or 'sensitivity' (aka. true positive rate)  = tp / (tp + fn) =", tp / (tp + fn))

print("Specificity             (aka. true negative rate)  = tn / (tn + fp) =", tn / (tn + fp))

print("Fall out                (aka. false positive rate) = fp / (fp + tn) =", fp / (fp + tn))

print("Miss rate               (aka. false negative rate) = fn / (fn + tp) =", fn / (fn + tp))
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve( perfect['Survived'] , gender['Survived'] )



from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score( perfect['Survived'] , gender['Survived'] )

print("AUC = ", roc_auc)



plt.figure(figsize = (8,5))

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.5f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic curve')

plt.legend(loc="lower right")

plt.show()
f1_score( perfect['Survived'] , gender['Survived'] )


accuracy_score( perfect['Survived'] , allZeros['Survived'] )

titanic_cm = confusion_matrix( perfect['Survived'] , allZeros['Survived'] )

print(titanic_cm)
# convert the ndarray to a pandas dataframe

cm_df = pd.DataFrame(titanic_cm)

# set the size of the figure

plt.figure(figsize = (5,5))

sn.heatmap(cm_df, 

           annot=True, annot_kws={"size": 25},

           fmt="d",         # decimals format

           xticklabels=False, 

           yticklabels=False,

           cmap="viridis", 

           cbar=False)

plt.show()
tn, fp, fn, tp = confusion_matrix( perfect['Survived'] , allZeros['Survived']).ravel()

print("Number of true negatives  (tn) = ",tn)

print("Number of true positives  (tp) = ",tp)

print("Number of false negatives (fn) = ",fn)

print("Number of false positives (fp) = ",fp)

print("Precision                                          = tp / (tp + fp) =", tp / (tp + fp))

print("Recall or 'sensitivity' (aka. true positive rate)  = tp / (tp + fn) =", tp / (tp + fn))

print("Specificity             (aka. true negative rate)  = tn / (tn + fp) =", tn / (tn + fp))

print("Fall out                (aka. false positive rate) = fp / (fp + tn) =", fp / (fp + tn))

print("Miss rate               (aka. false negative rate) = fn / (fn + tp) =", fn / (fn + tp))