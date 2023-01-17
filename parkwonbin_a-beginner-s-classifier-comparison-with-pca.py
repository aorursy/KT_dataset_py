import pandas as pd
train_set = pd.read_csv("../input/digit-recognizer/train.csv")

X = train_set.drop('label', axis=1)
y = train_set['label']
import numpy as np

# Split the train set into two.
num = int(len(X)*(3/5))
X_train, X_valid = X[:num],X[num:]
y_train, y_valid = y[:num],y[num:]

print("X_train :",len(X_train))
print("y_train :",len(y_train))
print("X_valid :",len(X_valid))
print("y_valid :",len(y_valid))
# Calculation takes too long, reduce size of data set
# X_train, X_valid = X_train[:500], X_valid[:200]
# y_train, y_valid = y_train[:500], y_valid[:200]

print("X_train :",len(X_train))
print("y_train :",len(y_train))
print("X_valid :",len(X_valid))
print("y_valid :",len(y_valid))
import time
import pandas as pd
from tqdm.notebook   import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

def train(*models,dataset=(X_train,y_train,X_valid, y_valid)):
    columns = ["Name", "Time(sec)","accuracy(%)", "precision(%)","recall(%)","f1-score","confusion" ,"model"]
    df = pd.DataFrame(columns=columns)
    
    X_train,y_train,X_valid,y_valid = dataset

    for model in tqdm(models) :
        model_name = str(model.__class__.__name__)
        print(model_name, end="...")
        
        # Time measurement
        start = time.time()
        
        # Trainning start
        model.fit(X_train,y_train)
        
        # report
        y_pred     = cross_val_predict(model, X_valid, y_valid, cv=3)     
        clf_report = classification_report(y_valid,y_pred, output_dict =True)
        
        accuracy   = clf_report["accuracy"]                # accuracy
        precision  = clf_report['macro avg']['precision']  # precision
        recall     = clf_report['macro avg']['recall']     # recall
        f1_score   = clf_report['macro avg']['f1-score']
        confusion  = confusion_matrix(y_valid, y_pred)     # confusion_matrix
        
        accuracy,precision,recall = [round(100*x,2) for x in [accuracy,precision,recall]]
        
        train_time = round(time.time() - start,2)

        # save data
        new_row = {f"{columns[0]}":model_name, # name
                   f"{columns[1]}":train_time, # training time
                   f"{columns[2]}":accuracy,   # accuracy
                   f"{columns[3]}":precision,  # precision
                   f"{columns[4]}":recall,     # recall 
                   f"{columns[5]}":f1_score,   # f1_score 
                   f"{columns[6]}":confusion,  # confusion_matrix 
                   f"{columns[7]}":model       # clf model
                  }
        
        df = df.append(new_row,ignore_index=True)    
        df = df.drop_duplicates(["Name"],keep="last")
        print("complite..!")
    return df
from sklearn.ensemble     import ExtraTreesClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.tree         import DecisionTreeClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm          import SVC
from sklearn.neighbors    import KNeighborsClassifier

# Random Seed
random_state = 20142927

# Definition of Classifiers
ext_clf = ExtraTreesClassifier(n_estimators=20,random_state=random_state)
det_clf = det_clf = DecisionTreeClassifier(splitter="random",criterion='entropy',random_state=random_state) # splitter="random" 빠름
rdf_clf = RandomForestClassifier(n_estimators=15, random_state=random_state)
knn_clf = KNeighborsClassifier(n_neighbors=20,leaf_size=50)
gnb_clf = GaussianNB()
log_clf = LogisticRegression()
sgd_clf = SGDClassifier(random_state=random_state)
svc_clf = SVC() 


# train and save classifiers
clf_data = train( 
    ext_clf, 
    det_clf, 
    rdf_clf, 
    knn_clf
)
from sklearn.ensemble     import VotingClassifier
from sklearn.ensemble     import BaggingClassifier
from sklearn.ensemble     import AdaBoostClassifier
from sklearn.ensemble     import GradientBoostingRegressor

bag_clf = BaggingClassifier(
    ExtraTreesClassifier(n_estimators=20,random_state=random_state),
    n_jobs=-1,
    n_estimators=5,
    random_state=random_state
)

ada_clf = AdaBoostClassifier(
    ExtraTreesClassifier(n_estimators=20,random_state=random_state), 
    n_estimators=50,
    learning_rate=0.2, 
    algorithm="SAMME.R", 
    random_state=random_state
)


vot_clf = VotingClassifier(
    estimators= [        
        ("ext_clf",ext_clf),
        ("rdf_clf",rdf_clf),
#         ("knn_clf",knn_clf), # Accurate, but takes long time
#         ("det_clf",det_clf), # The accuracy is too low
#         ("svc_clf",svm_clf), # The accuracy is too low
#         ("sgd_clf",sgd_clf), # Takes too much time
        ("bag_clf",bag_clf),
        ("ada_clf",ada_clf)
    ] , voting='soft'
)


clf_data = clf_data.append(
     train(bag_clf, ada_clf, vot_clf),ignore_index=True
)
clf_data.iloc[:,[0,1,2,5,6]]
for i in range(len(clf_data)) : 
    print(clf_data["Name"][i], end="\t")
    print(clf_data["accuracy(%)"][i], end="(%) \n  f1-score : ")
    print(clf_data["f1-score"][i],)
    print(clf_data["confusion"][i])
    print("\n")
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
%matplotlib inline

def show_auc(y_true,y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')    
    plt.legend(loc="lower right")
    
    
def Pretreatment_ROC(model, X, y):
    y_prob = model.predict_proba(X)
    y_pred = cross_val_predict(model, X, y)
    y_true = np.array(y == y_pred)
    y_score = [y_prob[i][y.iloc[i]] for i in range(len(y_prob))]
                        
    return y_true, y_score

    

def ROC_data(*models) :
    columns = ["Name","test_time","y_true","y_score"]
    df = pd.DataFrame()
    y_true, y_score =[], []
  
    for i in tqdm(range(len(models)))  :
        model = models[i]
        model_name = str(model.__class__.__name__) 
#         print(model_name, end="...")

       # Time measurement
        start = time.time()
        
        y_true, y_score = Pretreatment_ROC(model,X_valid,y_valid)
        
        test_time = round(time.time() - start,2)

        # data save
        new_row = {f"{columns[0]}":model_name,
                   f"{columns[1]}":test_time,  
                   f"{columns[2]}":y_true,  
                   f"{columns[3]}":y_score,  
                  }
        
        df = df.append(new_row,ignore_index=True)
        df = df.drop_duplicates(["Name"],keep="last")
#         print("complite..!")
    return df


def add_data(model,dataset):
    clf_data = train(model)
    ROC_dataset = ROC_data(model)
    new_row = pd.merge(clf_data, ROC_dataset, how='outer')
    return dataset.append(new_row)
models = [    
    ext_clf, 
    det_clf, 
    rdf_clf, 
    knn_clf, 
    bag_clf, 
    ada_clf, 
    vot_clf]


# Train and store data in 'dataset'
# clf_data = train(*models)
ROC_dataset = ROC_data(*models)
dataset = pd.merge(clf_data, ROC_dataset, how='outer')

dataset.iloc[:,[0,2,8,9,10]] 
plt.figure(figsize=(12,10))   
for i in tqdm(range(len(models)))  :
  
    # Positioning ==================
    col=2
    row = len(models)//col + 1
    plt.subplot(row,col,i+1)
    
    # plot ROC curve ===============
    y_true  = dataset.iloc[i,10]
    y_score = dataset.iloc[i,9]
    show_auc(y_true, y_score)
    
    # Get data ====================
    clf_name = dataset.iloc[i,0]
    accuracy = dataset.iloc[i,2]
    
    # Display Name and Accuracy =====
    tp = (0.2, 0.7) # Text Position
    text = "Accuracy : "+ str(accuracy)+"%"
    plt.text(tp[0],tp[1], clf_name, fontsize=15)
    plt.text(tp[0],tp[1]-0.1, text, fontsize=13)
    
plt.tight_layout()
plt.show()
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

class MNIST:        
    def __init__(self, X, y,pca):
        self.pca = pca
        self.X = X
        self.y = y
        self.ordinal = self.X
        self.reduced = self.pca.fit_transform(self.ordinal)
        self.recovered = self.pca.inverse_transform(self.reduced)
        
    def show(self,digit=None):
        if not digit : index = random.randint(0,len(self.X))
        else : index = int(digit)
        
        # Image preprocessing        
        image_ord = np.array(self.ordinal.iloc[index]).reshape(28, 28)
        image_rcd = np.array(self.recovered[index]).reshape(28, 28)
        
        # Plot Image
        plt.figure(figsize=(7, 4))
        pos = 121
        for img in [image_ord, image_rcd] :
            plt.subplot(pos)
            plt.title(f"y = {self.y[index]}",fontsize = 15)
            plt.imshow(img, cmap = matplotlib.cm.binary,interpolation="nearest")
            plt.axis("off"); pos += 1    
        plt.tight_layout()
        print(self.pca)
        
        
pca = PCA(n_components=0.8,whiten=True)
pca_train= MNIST(X_train,y_train,pca)
pca_valid = MNIST(X_valid,y_valid,pca)
pca_train.show()
models = [    
    ext_clf, 
    det_clf, 
    rdf_clf, 
    knn_clf, 
    bag_clf, 
    ada_clf, 
    vot_clf,
]


print("train ordinal data")
clf_ord = train(*models)

print("train redused data")
clf_pca = train(*models, dataset=(
    pca_train.reduced, pca_train.y,
    pca_valid.reduced,  pca_valid.y
))
clf_ord.iloc[:,[5]] =  round(100* clf_ord.iloc[:,[5]],2)
clf_ord_2 = clf_ord.iloc[:,[1,2,3,4,5]]
clf_ord_2.index = clf_ord["Name"]
clf_ord_2.columns = ["time","accuracy","percision","recall","f1-score"]

clf_pca.iloc[:,[5]] =  round(100* clf_pca.iloc[:,[5]],2)
clf_pca_2 = clf_pca.iloc[:,[1,2,3,4,5]]
clf_pca_2.index = clf_ord["Name"]
clf_pca_2.columns = ["time","accuracy","percision","recall","f1-score"]

clf_ord_2,clf_pca_2
label= [x.replace('Classifier', '') for x in clf_pca_2.index]

index = np.arange(len(label))

# plot Graph  ==================
for key in clf_pca_2.keys():
    plt.figure(figsize=(15,4))   
    w = 0.4
    plt.title(key, fontsize=15)

    plt.bar(index-w/2, clf_ord_2[key], width=w, label = "ord")
    plt.bar(index+w/2, clf_pca_2[key], width=w, label = "pca")
    
    # Axis setting
    y = [*clf_ord_2[key], *clf_pca_2[key]]
    plt.axis([-w, len(label)-1+w, min(y)*0.98, max(y)+3])
      
    # display Value
    for i in index:
        fs = 12
        dx = w*5/6
        dy = 0.8
        if key == "time" : d ="s";
        else : d = "%"
        plt.text(i-dx, clf_ord_2[key][i]+dy, str(clf_ord_2[key][i])+d,fontsize=fs)
        plt.text(i-dx+w, clf_pca_2[key][i]+dy, str(clf_pca_2[key][i])+d,fontsize=fs)

    # Displayed x-axis label
    plt.xticks(index, label, fontsize=14)
    plt.ylabel(key, fontsize=20)
    plt.legend(loc=6,fontsize=15)
    plt.show()
train = pd.read_csv("../input/digit-recognizer/train.csv")
test  = pd.read_csv("../input/digit-recognizer/test.csv")

X = train_set.drop('label', axis=1)
y = train_set['label']
def nameof(obj): return [name for name in globals()  if  globals()[name] is obj][0]


save = False
save = True

if save : 
    for i in range(len(models)) : 
        models[i].fit(X,y)
        models[i].predict(test)
        pd.DataFrame(models[i].predict(test)).to_csv(f"{nameof(models[i])}.csv")
else : 
    print("if you want to stroe result, set save to True")