#general
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time, datetime
import pickle
# EDA
from statsmodels.stats.outliers_influence import variance_inflation_factor
# ML libraries
## Preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn import decomposition
from imblearn.under_sampling import NearMiss

## Pipelines
from imblearn.pipeline import Pipeline # Pipeline for imbalanced dataset, oversampling ONLY in train, not in test
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline # another way of doing it, good for oversampling

## ML Algorithms
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC 
# !pip install --user annoy
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Metrics and visualizers
from sklearn.metrics import make_scorer, auc, accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve, plot_roc_curve, classification_report, confusion_matrix, plot_confusion_matrix, f1_score
import graphviz
# ML libraries
## FAISS Library, to speed knn --> Scikitlearn is too slow
# https://towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
# https://github.com/facebookresearch/faiss
# https://github.com/facebookresearch/faiss/issues/890
!apt -qq install -y libomp-dev
!pip install faiss
import faiss
# LOCAL MODULES
data_path = "../input/creditcardfraud/"
output_path = "./"
# GENERAL PURPOSE
RANDOM_STATE = 42

#The summary dictionary will include:
#- key: model's key
#- name: descriptive name of the model
#- model: python object with the estimator
#- accuracy: of the validation dataset
#- AUC: of the validation dataset
#- recall: of the validation dataset
#- precision: of the validation dataset
#- classification_report: of the validation dataset
summary = {}

# DATA
data_origin = "creditcard.csv"
# REPORTING Class
# This class is used to both compare train and test metrics of a model, and compare the different metrics
# of all the created models.
class CategorizationModelReport:

  model = None
  data = [
      # 0 - training
      {
        'X': None, 
        'y': None,
        'y_pred': None, 
      },
      # 1- validation
      {
        'X': None, 
        'y': None,
        'y_pred': None, 
      },
  ]
  metrics = [] # list storing dictionaries of metrics for train [0] and val [1]
  summary = {}

  # def __init__(self):

  def init_report(self, model, X_train, y_train, X_val, Y_val):  
    """Stores data and calculates the different metrics, stores it in the local vars and show them on screen"""

    self.model = model
    self.data[0]["X"] = X_train
    self.data[0]["y"] = y_train
    if "FaissKNeighbors" in str(type(model)):
      self.data[0]["y_pred"] = pd.Series(model.predict(np.ascontiguousarray(X_train.values)), index=X_train.index, name="Class_predicted")
      self.data[0]["y_pred_proba"] = None
    else:
      self.data[0]["y_pred"] = pd.Series(model.predict(X_train), index=X_train.index, name="Class_predicted")
      self.data[0]["y_pred_proba"] = pd.Series(pd.DataFrame(model.predict_proba(X_train), index=X_train.index).iloc[:,-1], name="Class_predicted")
    self.data[1]["X"] = X_val
    self.data[1]["y"] = y_val
    if "FaissKNeighbors" in str(type(model)):
      self.data[1]["y_pred"] = pd.Series(model.predict(np.ascontiguousarray(X_val.values)), index=X_val.index, name="Class_predicted")
      self.data[1]["y_pred_proba"] = None
    else:
      self.data[1]["y_pred"] = pd.Series(model.predict(X_val), index=X_val.index, name="Class_predicted")
      self.data[1]["y_pred_proba"] = pd.Series(pd.DataFrame(model.predict_proba(X_val), index=X_val.index).iloc[:,-1], name="Class_predicted")
    
    # Calculate the metrics
    self.metrics = [self.empty_metrics(), self.empty_metrics()] # train and validation
    for i in range(0, len(self.metrics)):
      self.metrics[i]["accuracy"] = round(accuracy_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
      self.metrics[i]["precision"] = round(precision_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
      self.metrics[i]["recall"] = round(recall_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
      self.metrics[i]["f1"] = round(f1_score(self.data[i]["y"], self.data[i]["y_pred"]),4)
      self.metrics[i]["cm"] = confusion_matrix(self.data[i]["y"], self.data[i]["y_pred"])
      if "FaissKNeighbors" not in str(type(model)):
        self.metrics[i]["auc"] = round(roc_auc_score(self.data[i]["y"], self.data[i]["y_pred_proba"]),4)
        self.metrics[i]["tpr"], self.metrics[i]["fpr"], thresholds = roc_curve(self.data[i]["y"], self.data[i]["y_pred_proba"])
  
  def show_report(self):
    '''Shows the report for this model'''
    
    display(pd.DataFrame(self.metrics, index=["train","validation"])[["accuracy","precision","recall","f1","auc"]])
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    axs[0].title.set_text("Train Confusion Matrix")
    sns.heatmap(self.metrics[0]["cm"], annot=True, annot_kws={"size":9}, fmt='g', ax=axs[0]) # fmt = format of annot, in that case, plain notation (g)
    axs[1].title.set_text("Validation Confusion Matrix")
    sns.heatmap(self.metrics[1]["cm"], annot=True, annot_kws={"size":9}, fmt='g', ax=axs[1]) # fmt = format of annot, in that case, plain notation (g)
    plt.show()

    if "FaissKNeighbors" not in str(type(model)):
      fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
      axs[0].title.set_text("Train ROC Curve")
      plot_roc_curve(self.model, self.data[0]["X"], self.data[0]["y"], ax=axs[0])
      axs[1].title.set_text("Validation ROC Curve")
      plot_roc_curve(self.model, self.data[1]["X"], self.data[1]["y"], ax=axs[1])

  def plot_roc_curve_custom(self, y = None, y_pred_prob = None): # deprecated
    """Plots the roc curve of the trained model"""
    if y is None: 
      y, y_pred_prob = self.y, self.y_pred_prob
    
    fpr, tpr, threshold = roc_curve(y, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.margins(0.1)
    plt.show()
  
  def show_cv_results(self, cv_results, metric):
    '''
    Compares the results of a GridSearchCV

    parameters:
    - cv_results -> the cv_results_ object stored in the GridSearchCV instance (GridSearchCV.cv_results_)
    - metric -> the name of the metric to compare
    '''
    # The detailed results of the Cross-Validation process are stored the follwing way
    # cv_results_ is a dictionary
    # Metric results are stored for each Kfold split in the following keys: 
    # - split*number_of_fold*_test_*metric*
    # - split*number_of_fold*_train_*metric*
    # For each kfold split there is an array with length j. The length j corresponds to all combinations of the parameters fed to GridSearchCV
    # The different combinations of hypermparameters are stored in this array, from 0 to j-1

    print ("********************************")
    print ("METRIC: {}".format(metric.upper()))
    print ("********************************")

    fig, axes = plt.subplots(1, len(cv_results["params"]), sharey = True, figsize=(5*len(cv_results["params"]), 5))
    fig.subplots_adjust(hspace=0.3)
    # plt.xticks(np.arange(gscv.n_splits_), np.arange(1,gscv.n_splits_+1))

    # We gather the best model that GridSearchCV has calculated
    # e.g. 'rank_test_accuracy': array([2, 1, 3], dtype=int32)
    iModel_best = cv_results["rank_test_" + metric].tolist().index(1) # 1 is the first position in ranking, the best model in test dataset

    # We loop through the hyperparameters combination
    models_scores = pd.DataFrame(columns=["model","split","train","test"])
    for iModel, model in enumerate(cv_results["params"]): # params is an array of dictionaries
      split_metric_scores = None # this list will store the train and test metric scores for each of the splits
      has_metrics = True
      for iSplit in range(0, gscv.n_splits_):
        # if we find that there are no values, we must inform the result as nan
        if np.isnan(cv_results["split" + str(iSplit) + "_train_" + metric][iModel]):
          has_metrics = False
        if has_metrics:
          # store the metric
          split_metric_scores = pd.DataFrame([[
            iModel, 
            int(iSplit),
            cv_results["split" + str(iSplit) + "_train_" + metric][iModel], 
            cv_results["split" + str(iSplit) + "_test_" + metric][iModel]
          ]], columns=models_scores.columns)
        else:
          # store nan
          split_metric_scores = pd.DataFrame([[
            iModel, 
            int(iSplit),
            np.nan, 
            np.nan
          ]], columns=models_scores.columns)
        models_scores = models_scores.append(split_metric_scores, ignore_index=True)
      # and we plot the values for train and test for the different splits in the model
      axes[iModel].title.set_text("MODEL {}".format(iModel))
      if iModel == iModel_best: axes[iModel].set_facecolor('wheat') #highlight the selected model by GridSearchCV
      if has_metrics: sns.lineplot(data=models_scores[models_scores["model"]==iModel].iloc[:,1:].set_index("split"), ax=axes[iModel])
    plt.show()

    # And one last plot to compare the models
    fig, axes = plt.subplots(1, 2, sharey = True, sharex = True, figsize=(10,5))
    fig.subplots_adjust(wspace=0.3)
    # plt.xticks(ticks=np.arange(len(cv_results["params"])), labels=np.arange(1,len(cv_results["params"])+1))
    axes[0].title.set_text("MODEL COMPARISON - Train")
    sns.boxplot(data=models_scores, x="model", y="train", ax=axes[0], showmeans=True)
    #sns.swarmplot(data=models_scores, x="model", y="train", color=".25", ax=axes[0])
    axes[1].title.set_text("MODEL COMPARISON - Test")
    sns.boxplot(data=models_scores, x="model", y="test", ax=axes[1], showmeans=True)
    #sns.swarmplot(data=models_scores, x="model", y="test", color=".25", ax=axes[1])
    plt.show()
  
  def get_model_params(self, cv_results, model_key):
    '''
    Returns the params of the model using the key in the GridSearchCV cv_results_ object.

    parameters:
    cv_results -> GridSearchCV.cv_results_
    model_key -> the key of the selected model on cv_results["params"]

    returns:
    dictionary with the model parameters
    '''
    return {model_key:cv_results["params"][model_key]}
  
  def store_report(self, key = "", name = ""):
    """Add the VALIDATION metrics of the current model to the dictionary of compared models"""
    
    self.summary[key] = {
      "name" : name,
      "model" : self.model, 
      "y": self.data[1]["y"], 
      "y_pred": self.data[1]["y_pred"], 
      "y_pred_proba": self.data[1]["y_pred_proba"], 
      "accuracy" : self.metrics[1]["accuracy"], 
      "precision": self.metrics[1]["precision"], 
      "recall": self.metrics[1]["recall"], 
      "f1": self.metrics[1]["f1"], 
      "auc" : self.metrics[1]["auc"], 
      "tpr": self.metrics[1]["tpr"], 
      "fpr": self.metrics[1]["fpr"], 
      "cm": self.metrics[1]["cm"], 
    }
  
  def compare_models(self, order_by="accuracy"):
    """Plots the metrics of the different models that have been stored"""
    df_summary = pd.DataFrame(self.summary).T.reset_index()
    return df_summary[["name","accuracy","recall","precision","f1","auc"]].sort_values(by=order_by, ascending=False)
  
  def empty_metrics(self):
    return {
      'accuracy': None, 
      'precision': None, 
      'recall': None, 
      'f1': None, 
      'cm': None, 
      'auc': None, 
      'fpr': None, 
      'tpr': None, 
    }

report = CategorizationModelReport()
# FAISS Class to execute Knn algorithms
# https://github.com/j-adamczyk/Towards_Data_Science/issues/1 --> issues with series, dataframes and numpy arrays solved
class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self._y_np = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y
        self._y_np = np.array(y, dtype=np.int)

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        # votes = self.y[indices]
        votes = self._y_np[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions
data = pd.read_csv(data_path + data_origin)

# to visualize pairplots and correlations, we will reduce the data and balance the class
data_ = data[data["Class"]==1]
data_ = data_.append(data[data["Class"]==0].sample(n=len(data_), random_state=RANDOM_STATE))

# Correlation matrix
corr_ = data_.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(25, 25))

# Generate a custom diverging colormap
#sns.set_theme(style="white")
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_, annot=True, mask=mask, cmap=cmap, vmax=1, center=0,
          square=True, linewidths=.5, cbar_kws={"shrink": .5})

# We finally print boxplots for all variables
fig, axes = plt.subplots(6, 5, figsize=(15,25), sharex = True)
fig.tight_layout(pad=2.0)
cols = data.columns.drop("Class")
for i in range(0, len(cols)):
    col = (i % 5)
    if col == 0: row = int(i/5) # change row
    sns.boxplot(data=data_, x="Class", y=cols[i], ax=axes[row, col], whis=4) # instead of 1.5IQR we loog for extreme outliers, over 3IQR
plt.show()

# Calculating VIF
vif = pd.DataFrame()
vif["variables"] = data.columns[:-1]
vif["VIF"] = [variance_inflation_factor(data.iloc[:,:-1].values, i) for i in range(data.shape[1]-1)]
display(vif)
# The variables come from PCA, except from Time and Amount. We will Scale these two variables in order
# to have everything at the same level of information
scaler = StandardScaler()
scaler.fit(data[["Time","Amount"]])
scaled_cols = scaler.transform(data[["Time","Amount"]])
scaled_cols
df_ = pd.DataFrame(scaled_cols)
data["time_scaled"] = df_[0]
data["amount_scaled"] = df_[1]
data.drop(["Time","Amount"], axis=1, inplace=True)
data
del scaled_cols
# Create X and y
X_data = data.drop("Class", axis=1)
y_data = data["Class"]
# get dev and val
# val is our FINAL VALIDATION DATASET, AND IT WILL BE THE SAME TO COMPARE THE DIFFERENT MODELS
X_dev, X_val, y_dev, y_val = train_test_split(X_data, y_data, 
                                                    train_size=396/492, 
                                                    random_state=RANDOM_STATE,
                                                    stratify=y_data)
# Let's check the target distribution
display('dev dataset: {}'.format(y_dev.sum()/len(y_dev)))
display('val dataset: {}'.format(y_val.sum()/len(y_val)))
X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, 
                                                    train_size=300/396, 
                                                    random_state=RANDOM_STATE,
                                                    stratify=y_dev)
# Let's check the target distribution
display('train dataset: {}'.format(y_train.sum()/len(y_train)))
display('test dataset: {}'.format(y_test.sum()/len(y_test)))
print("Data length: {}".format(len(data)))
print("Dev length: {}".format(len(X_dev)))
print("Train length: {}".format(len(X_train)))
print("Test length: {}".format(len(X_test)))
print("Validation length: {}".format(len(X_val)))
# Check for number of positive cases
print("Number of positive classes = {}".format(y_data.sum()))
# Instantiate and fit nearmiss object
nm = NearMiss()

# We will undersample the dev dataset, as validation must be kept as originally generated for comparison purposes
X_dev_us, y_dev_us = nm.fit_resample(X_dev, y_dev)
X_dev_us = pd.DataFrame(X_dev_us, columns=X_dev.columns)

print("New dataset length = {}".format(len(y_dev_us)))
print("New dataset class balance = {}".format(y_dev_us.sum()/len(y_dev_us)))
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_dev_us, y_dev_us, 
                                                    train_size=300/396, 
                                                    random_state=RANDOM_STATE,
                                                    stratify=y_dev_us)
# Let's check the target distribution
display('train dataset: {}'.format(y_train_us.sum()/len(y_train_us)))
display('test dataset: {}'.format(y_test_us.sum()/len(y_test_us)))
print("Dev length: {}".format(len(X_dev_us)))
print("Train length: {}".format(len(X_train_us)))
print("Test length: {}".format(len(X_test_us)))
# Let's check the number of positive class entries that exist and the corresponding length of the oversampling
print("Number of positive classes = {}".format(len(y_dev[y_dev==0])))
smoter = SMOTE(random_state=RANDOM_STATE)

# We oversample the dev dataset 
X_dev_os, y_dev_os = smoter.fit_resample(X_dev, y_dev)
X_dev_os = pd.DataFrame(X_dev_os, columns=X_dev.columns)
X_dev_os_fcols = pd.DataFrame(X_dev_os, columns=["f" + str(x) for x in range(0, len(X_dev.columns))])

# We oversample the train dataset
X_train_os, y_train_os = smoter.fit_resample(X_train, y_train)
X_train_os = pd.DataFrame(X_train_os, columns=X_train.columns)
X_train_os_fcols = pd.DataFrame(X_train_os, columns=["f" + str(x) for x in range(0, len(X_train.columns))])

# We change the column names to dev dataset
X_dev_fcols = X_dev.copy()
X_dev_fcols.columns = ["f" + str(x) for x in range(0, len(X_dev.columns))]

# We change the column names to test dataset
X_test_fcols = X_test.copy()
X_test_fcols.columns = ["f" + str(x) for x in range(0, len(X_test.columns))]

# We change the column names to dev dataset
X_val_fcols = X_val.copy()
X_val_fcols.columns = ["f" + str(x) for x in range(0, len(X_val.columns))]
print("Length of oversampled training set = {}".format(len(X_train_os)))
print("Balance of oversampled training set Class = {}".format(y_train_os.sum()/len(y_train_os)))
print("Dev length: {}".format(len(X_dev_os)))
print("Train length: {}".format(len(X_train_os)))
# We will loop on different max_depths to get the optimal result
for i in range(3, 7):
    dt = DecisionTreeClassifier(max_depth=i, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    
    y_train_prediction = dt.predict(X_train)
    y_test_prediction = dt.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_train_prediction)
    test_accuracy = accuracy_score(y_test, y_test_prediction)
    print("Tree depth: {}. AUC --> Train: {} - Test: {}".format(i, round(train_accuracy,5), round(test_accuracy,5)))
    train_f1 = f1_score(y_train, y_train_prediction)
    test_f1 = f1_score(y_test, y_test_prediction)
    print("               F1  --> Train: {} - Test: {}".format(round(train_f1,5), round(test_f1,5)))
    print("")
dt = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
dt.fit(X_dev, y_dev)
model = dt
# REPORT
report.init_report(model, X_dev, y_dev, X_val, y_val)
report.show_report()
report.store_report(
  key="dt_unbalanced", 
  name="Decision Tree - Unbalanced Baseline"
)
report.compare_models()
# Set the kfold strategy
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) # we will keep 5 splits in order to keep a significant number of frauds in the positive class

# Instanciate the model
xgb = XGBClassifier()

# Set parameters for GridSearchCV. We will only provide different options for the max_depth
parameters = {
  'objective':['binary:logistic'],
  'learning_rate': [0.05],
  'max_depth': list(range(3, 7)),
  #'min_child_weight': [11],
  #'silent': [1],
  #'subsample': [0.7],
  #'colsample_bytree': [0.7],
  'n_estimators': [10], #number of trees
  #'missing':[-999], # we have no missings in the db
  'seed': [RANDOM_STATE]
}

# Instantiate Grid Search
gscv = GridSearchCV(
    xgb, 
    parameters, 
    cv = kfold, 
    scoring = ['accuracy','precision','recall','roc_auc','f1'], 
    return_train_score = True, # we want to see the difference between train and test 
    verbose = 1, 
    refit = 'accuracy' 
)

gscv.fit(X_dev, y_dev)
report.show_cv_results(gscv.cv_results_, "accuracy")
report.get_model_params(gscv.cv_results_, 1)
# Let's check all the scores before proceeding with the best estimator
display("mean_train_accuracy {}".format(gscv.cv_results_["mean_train_accuracy"]))
display("mean_test_accuracy {}".format(gscv.cv_results_["mean_test_accuracy"]))
display("mean_train_roc_auc {}".format(gscv.cv_results_["mean_train_roc_auc"]))
display("mean_test_roc_auc {}".format(gscv.cv_results_["mean_test_roc_auc"]))
display('Accuracy best score: {}'.format(gscv.best_score_))
# Let's check the score for the validation partition
display("Score {}".format(gscv.score(X_val, y_val)))
# Let's paint the ROC curve
display(plot_roc_curve(gscv.best_estimator_, X_dev, y_dev))
# SAVE best model
model = gscv.best_estimator_
# REPORT
report.init_report(model, X_dev, y_dev, X_val, y_val)
report.show_report()
report.store_report(
  key="xgb_unbalanced_gridsearchcv", 
  name="XGBClassifier - Unbalanced with GridSearchCV"
)
report.compare_models()
# We have to work with the dev dataset. When oversampling, we should be carefull, 
# as we ONLY want to oversample the TRAINING data, not the TEST.
# We will use a specific pipeline (imblearn) to handle this situation

logreg = LogisticRegression()

parameters = {
    'penalty' : ['l1','l2'],
    'max_iter' : [100, 500, 1000], 
    'solver' : ['liblinear'], # better for small datasets, but it can be used for both l1 and l2 penalty
  }

# Instantiate Grid Search
gscv = GridSearchCV(
    logreg, 
    param_grid = parameters, 
    scoring = ['accuracy','roc_auc'], 
    cv = 5, # default value
    return_train_score = True, # we want to see the difference between train and test
    verbose = 2, 
    refit = 'accuracy' # yes, using accuracy as the best estimator metric
)

gscv.fit(X_dev, y_dev) # we include dev, as the GridSearchCV will perform k-fold strategy
report.show_cv_results(gscv.cv_results_, "accuracy")
# Let's check all the scores before proceeding with the best estimator
display("mean_train_accuracy {}".format(gscv.cv_results_["mean_train_accuracy"]))
display("mean_test_accuracy {}".format(gscv.cv_results_["mean_test_accuracy"]))
display("mean_train_roc_auc {}".format(gscv.cv_results_["mean_train_roc_auc"]))
display("mean_test_roc_auc {}".format(gscv.cv_results_["mean_test_roc_auc"]))
display('Accuracy best score: {}'.format(gscv.best_score_))
# Let's paint the ROC curve
display(plot_roc_curve(gscv.best_estimator_, X_dev, y_dev))
gscv.best_params_
# SAVE best model
model = gscv.best_estimator_
# REPORT
report.init_report(model, X_dev, y_dev, X_val, y_val)
report.show_report()  
report.store_report(
  key="logreg_unbalanced_gridsearchcv", 
  name="Logistic Regression - Unbalanced with GridSearchCV"
)
report.compare_models()
knn_faiss = FaissKNeighbors()
knn_faiss.fit(np.ascontiguousarray(X_dev.values), y_dev)
model = knn_faiss
report.init_report(model, X_dev, y_dev, X_val, y_val)
report.show_report()
report.store_report(
  key="knnfaiss_unbalanced_k5", 
  name="Knn Faiss - Unbalanced"
)
report.compare_models()
svc = LinearSVC()

parameters = {
    'penalty' : ['l2'], 
    'C' : [0.1, 1], # [0.1, 1, 2, 5, 10, 25, 50, 100], 
    'random_state' : [RANDOM_STATE], 
    'max_iter' : [100, 1000]
  }

# Instantiate Grid Search
gscv = GridSearchCV(
    svc, 
    param_grid = parameters, 
    scoring = ['accuracy','roc_auc'], 
    cv = 5, # default value, c
    return_train_score = True, # we want to see the difference between train and test
    verbose = 2, 
    refit = 'accuracy' # yes, using accuracy as the best estimator metric
)

gscv.fit(X_dev, y_dev) # we include dev, as the GridSearchCV will perform k-fold strategy
report.show_cv_results(gscv.cv_results_, "accuracy")
# Let's check all the scores before proceeding with the best estimator
display("mean_train_accuracy {}".format(gscv.cv_results_["mean_train_accuracy"]))
display("mean_test_accuracy {}".format(gscv.cv_results_["mean_test_accuracy"]))
display("mean_train_roc_auc {}".format(gscv.cv_results_["mean_train_roc_auc"]))
display("mean_test_roc_auc {}".format(gscv.cv_results_["mean_test_roc_auc"]))
display('Accuracy best score: {}'.format(gscv.best_score_))
# Let's paint the ROC curve
display(plot_roc_curve(gscv.best_estimator_, X_dev, y_dev))
plt.show()
gscv.best_params_
# RETRAIN THE MODEL and get 
svc = LinearSVC(
    penalty = "l2", 
    C = 1,
    max_iter = 100, 
    random_state = RANDOM_STATE, 
)
svc.fit(X_dev, y_dev)
# We use CalibratedClassifier to get prediction probabilites, as LinearSVC does not natively include them
clf = CalibratedClassifierCV(
  svc, 
  cv = "prefit",  # the model is already fit, no need to do CV
)
clf.fit(X_val, y_val)
model = clf
report.init_report(clf, X_dev, y_dev, X_val, y_val)
report.show_report()
report.store_report(
  key="svc_unbalanced_gridsearchcv", 
  name="SVC - Unbalanced with GridSearchCV"
)
report.compare_models()
# We have to work with the dev dataset. When oversampling, we should be carefull, 
# as we ONLY want to oversample the DEV/TRAINING data, not the TEST neither VALIDATION.
# We will use a specific pipeline (imblearn) to handle this situation

xgb = Pipeline([
        ('sampling', SMOTE()),
        ('classification', XGBClassifier())
])

parameters = {
    'classification__objective':['binary:logistic'],
    'classification__learning_rate': [0.05],
    'classification__max_depth': list(range(3, 7)),
    #'min_child_weight': [11],
    #'silent': [1],
    #'subsample': [0.7],
    #'colsample_bytree': [0.7],
    'classification__n_estimators': [10], #number of trees
    #'missing':[-999], # we have no missings in the db
    'classification__seed': [RANDOM_STATE]
  }
# Instantiate Grid Search
gscv = GridSearchCV(
    xgb, 
    param_grid = parameters, 
    cv = 5, # default is 5
    scoring = ['accuracy','precision','recall','roc_auc','f1'], 
    return_train_score = True, # we want to see the difference between train and test
    verbose = 2, 
    refit = 'accuracy' # although we will refit manually
)

gscv.fit(X_dev_fcols, y_dev)
report.show_cv_results(gscv.cv_results_, "accuracy")
max_depth = gscv.best_params_["classification__max_depth"]
print("Optimal max_depth: {}".format(max_depth))
# Let's check all the scores before proceeding with the best estimator
display("mean_train_accuracy {}".format(gscv.cv_results_["mean_train_accuracy"]))
display("mean_test_accuracy {}".format(gscv.cv_results_["mean_test_accuracy"]))
display("mean_train_roc_auc {}".format(gscv.cv_results_["mean_train_roc_auc"]))
display("mean_test_roc_auc {}".format(gscv.cv_results_["mean_test_roc_auc"]))
display('AUC best score: {}'.format(gscv.best_score_))
# SAVE best model
model = XGBClassifier(
    objective = "binary:logistic",
    learning_rate = 0.05,
    max_depth = max_depth,
    n_estimators = 10, # number of trees
    seed = RANDOM_STATE, 
)
# We fit the model with the best parameters. We do it on the dev data (oversampled)
model.fit(X_dev_os, y_dev_os)
report.init_report(model, X_dev, y_dev, X_val, y_val)
report.show_report()
report.store_report(
  key="xgb_oversamplig_gridsearchcv",
  name="XGBClassifier - Oversampling with GridSearchCV"
)
report.compare_models()
# METRICS - We will store all the metrics in a dataframe
df_results_ = pd.DataFrame(columns=["depth","iter","error_train","error_test","accuracy_train","accuracy_test"]) # when iter = Nan, that means that we are averaging all of them

# DEPTH LOOP - As we will not be using GridSearchCV, we will iterate different depths to select the best model
min_depth = 4
max_depth = 7
for idepth in range(min_depth, max_depth+1):

  # Initialize xgbclassifier
  xgboost = XGBClassifier(max_depth=idepth, n_estimators=10, seed=RANDOM_STATE)

  # We create the kfold splitter. We will always shuffle the results. We use Stratified to ensure the ratio of positive class between splits
  kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) # we will keep 5 splits in order to keep a significant number of frauds in the positive class

  # We now loop through the kfolds
  i = 0
  for train_indexes_, test_indexes_ in kfold.split(X_dev, y_dev): # split generates two arrays of indexes, one for train and the other for test
    print("Depth: {} - KFold: {}".format(idepth, i+1))
    # We now generate the datasets from the indexes
    X_train_, y_train_ = X_dev.iloc[train_indexes_], y_dev.iloc[train_indexes_]
    X_test_, y_test_ = X_dev.iloc[test_indexes_], y_dev.iloc[test_indexes_]
    
    # We need to oversample ONLY the training set using SMOTE
    smoter = SMOTE(random_state=RANDOM_STATE)
    X_train_, y_train_ = smoter.fit_resample(X_train_, y_train_)
    X_train_, y_train_ = pd.DataFrame(X_train_, columns=X_dev.columns), pd.Series(y_train_)
    
    # We train the model and test at the same time
    xgboost.fit(X_train_, y_train_, eval_set=[(X_train_, y_train_), (X_test_, y_test_)], eval_metric="error", verbose=False)

    # And finally we store the results for this loop and calculate the average
    evals_result = xgboost.evals_result()
    result_ = [idepth, i+1, np.mean(evals_result["validation_0"]["error"]), np.mean(evals_result["validation_1"]["error"]), 1-np.mean(evals_result["validation_0"]["error"]), 1-np.mean(evals_result["validation_1"]["error"])]
    df_results_ = df_results_.append(pd.Series(result_, index = df_results_.columns), ignore_index=True)
    i += 1

  depth_average_ = df_results_[df_results_["depth"]==idepth].groupby(by=["depth"], as_index=False).mean()
  depth_average_.loc[0,"iter"] = np.nan
  df_results_ = df_results_.append(depth_average_, ignore_index=True)
  
display(df_results_)


fig, axs = plt.subplots(max_depth - min_depth + 1, sharey=True, sharex=True, figsize=(15,25))
axs[0].set(ylim=(0.94, 1.0))
plt.xticks(ticks=range(1,6))
i = 0
for idepth in range(min_depth, max_depth+1):
  axs[i].set_title("Depth: " + str(idepth))
  sns.lineplot(
      data=df_results_[(df_results_["depth"]==idepth) & (~df_results_["iter"].isnull())], 
      x="iter", 
      y="accuracy_train",
      color="b",
      ax = axs[i]
  )

  sns.lineplot(
      data=df_results_[(df_results_["depth"]==idepth) & (~df_results_["iter"].isnull())], 
      x="iter", 
      y="accuracy_test",
      color="g",
      ax = axs[i]
  )


  # Average for train
  acc_ = df_results_[(df_results_["depth"]==idepth) & (df_results_["iter"].isnull())]["accuracy_train"].iloc[0]
  axs[i].axhline(acc_, ls='--', color='b')
  # Average for test
  acc_ = df_results_[(df_results_["depth"]==idepth) & (df_results_["iter"].isnull())]["accuracy_test"].iloc[0]
  axs[i].axhline(acc_, ls='-.', color='g')

  i+=1
# Initialize xgbclassifier
xgb = XGBClassifier(
  max_depth=7, 
  n_estimators=10, 
  seed=RANDOM_STATE
)
  
# We train the model and validate at the same time
# eval_metric = error = #(wrong cases)/#(all cases) (1-accuracy)
# The training dataset is the oversampled one.
xgb.fit(X_dev_os, y_dev_os, eval_set=[(X_dev_os, y_dev_os), (X_val, y_val)], eval_metric="error", verbose=False)

# And finally we store the results for this loop and calculate the average
evals_result = xgb.evals_result()
evals_result
# SAVE best model
model = xgb
report.init_report(model, X_dev_os, y_dev_os, X_val, y_val)
report.show_report()
report.store_report(
  key="xgb_oversamplig_kfold",
  name="XGBClassifier - Oversampling with Kfold"
)
report.compare_models()
# We have to work with the dev dataset. When oversampling, we should be carefull, 
# as we ONLY want to oversample the TRAINING data, not the TEST.
# We will use a specific pipeline (imblearn) to handle this situation

logreg = Pipeline([
        ('sampling', SMOTE()),
        ('classification', LogisticRegression())
])

parameters = {
    'classification__penalty' : ['l1','l2'],
    'classification__max_iter' : [100, 500, 1000], 
    'classification__solver' : ['liblinear'], 
    'classification__random_state': [RANDOM_STATE],
}
# Instantiate Grid Search
gscv = GridSearchCV(
    logreg, 
    param_grid = parameters, 
    scoring = ['accuracy','roc_auc'], 
    return_train_score = True, # we want to see the difference between train and test
    verbose = 2, 
    refit = 'accuracy' 
)

gscv.fit(X_dev_fcols, y_dev)
report.show_cv_results(gscv.cv_results_, "accuracy")
# SAVE best model
params = gscv.best_params_
params
# Instantiate model
logreg = LogisticRegression(
    penalty = params["classification__penalty"],
    random_state = RANDOM_STATE, 
    max_iter = params["classification__max_iter"],  
    solver = 'liblinear', 
)
logreg.fit(X_dev_os, y_dev_os)
model = logreg
report.init_report(model, X_dev_os, y_dev_os, X_val, y_val)
report.show_report()
report.store_report(
  key="logreg_oversamplig_gridsearchcv",
  name="Logistic Regression - Oversampling with GridSearch"
)
report.compare_models()
# We have to work with the dev dataset. When oversampling, we should be carefull, 
# as we ONLY want to oversample the DEV/TRAINING data, not the TEST neither VALIDATION.
# We will use a specific pipeline (imblearn) to handle this situation

xgb = XGBClassifier(
)

parameters = {
    'objective':['binary:logistic'],
    'learning_rate': [0.05],
    'max_depth': list(range(4, 7)),
    'n_estimators': [10], #number of trees
    'seed': [RANDOM_STATE]
  }
# Instantiate Grid Search
gscv = GridSearchCV(
    xgb, 
    param_grid = parameters, 
    cv = 5, # default is 5
    scoring = ['accuracy','precision','recall','roc_auc','f1'], 
    return_train_score = True, # we want to see the difference between train and test
    verbose = 2, 
    refit = 'accuracy' # although we will refit manually
)

gscv.fit(X_dev_us, y_dev_us)
report.show_cv_results(gscv.cv_results_, "accuracy")
max_depth = gscv.best_params_["max_depth"]
print("Optimal max_depth: {}".format(max_depth))
# Let's check all the scores before proceeding with the best estimator
display("mean_train_accuracy {}".format(gscv.cv_results_["mean_train_accuracy"]))
display("mean_test_accuracy {}".format(gscv.cv_results_["mean_test_accuracy"]))
display("mean_train_roc_auc {}".format(gscv.cv_results_["mean_train_roc_auc"]))
display("mean_test_roc_auc {}".format(gscv.cv_results_["mean_test_roc_auc"]))
display('AUC best score: {}'.format(gscv.best_score_))
# SAVE best model
model = XGBClassifier(
    objective = "binary:logistic",
    learning_rate = 0.05,
    max_depth = max_depth,
    n_estimators = 10, # number of trees
    seed = RANDOM_STATE, 
)
# We fit the model with the best parameters. We do it on the dev data (oversampled)
model.fit(X_dev_us, y_dev_us)
report.init_report(model, X_dev_us, y_dev_us, X_val, y_val)
report.show_report()
report.store_report(
  key="xgb_undersamplig_gridsearchcv",
  name="XGBClassifier - Undersampling with GridSearchCV"
)
report.compare_models()
# We have to work with the dev dataset. When oversampling, we should be carefull, 
# as we ONLY want to oversample the TRAINING data, not the TEST.
# We will use a specific pipeline (imblearn) to handle this situation

logreg = LogisticRegression()

parameters = {
    'penalty' : ['l1','l2'],
    'max_iter' : [100, 500, 1000], 
    'solver' : ['liblinear'], 
    'random_state': [RANDOM_STATE],
}
# Instantiate Grid Search
gscv = GridSearchCV(
    logreg, 
    param_grid = parameters, 
    scoring = ['accuracy','roc_auc'], 
    return_train_score = True, # we want to see the difference between train and test
    verbose = 2, 
    refit = 'accuracy' 
)

gscv.fit(X_dev_us, y_dev_us)
report.show_cv_results(gscv.cv_results_, "accuracy")
# SAVE best model
params = gscv.best_params_
params
# Instantiate model
logreg = LogisticRegression(
    penalty = params["penalty"],
    random_state = RANDOM_STATE, 
    max_iter = params["max_iter"],  
    solver = 'liblinear', 
)
logreg.fit(X_dev_us, y_dev_us)
model = logreg
report.init_report(model, X_dev_us, y_dev_us, X_val, y_val)
report.show_report()
report.store_report(
  key="logreg_undersamplig_gridsearchcv",
  name="Logistic Regression - Undersampling with GridSearch"
)
report.compare_models()
report.compare_models()
# And finally, we generate the report and save the Report instance to have all the info with us
version = str(datetime.datetime.now().year) + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + "_" + str(datetime.datetime.now().hour) + "_" + str(datetime.datetime.now().minute)
# CSV file with the results
report.compare_models().to_csv(output_path + version + " - Credit Card Fraud Detection - report.csv")
# Pickle with the report
pickle.dump(report, open( output_path + version + "Credit Card Fraud Detection - report pickle.pkl", "wb" ))