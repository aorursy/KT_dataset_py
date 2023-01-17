import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from mlxtend.classifier import StackingClassifier

import matplotlib.pyplot as plt 

import matplotlib.font_manager as font_manager

from matplotlib import rcParams

import seaborn as sns

from matplotlib.pyplot import figure
import os

print(os.listdir('../input/'))

data = pd.read_csv('../input/data_main.csv')
data
dataframe_licht = []
metrics = pd.DataFrame(index = ['f1_score', 'precision_score', 'recall_score','accuracy'], 

          columns = ["class_0_and_1", "class_2_and_3", "class_4_and_5", "class_6_and_7"])

class_name = ["class_0_and_1", "class_2_and_3", "class_4_and_5", "class_6_and_7"]
def predict_uith_metric(name,feature, target):

    

    

    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.33)

    m = StackingClassifier(classifiers=[LogisticRegression(),XGBClassifier(max_depth=2)],use_probas=True, meta_classifier=LogisticRegression())

    m.fit(X_train, y_train)

    

    preds = pd.DataFrame()

    preds['stack_pred'] = m.predict(X_test)

    print(preds.head())

    y_hat = preds.iloc[0:,0]

    

    accuracy = accuracy_score(y_test, y_hat)

    print(accuracy)

    print(classification_report(y_test, y_hat))

    classification_report_data = classification_report(y_test, y_hat, output_dict=True)

    

    classification_report_df = pd.DataFrame(classification_report_data).transpose()

    classification_report_df.to_csv(name+'.csv', index= True)

    

    precision_weighted = classification_report_df["precision"][classification_report_df.index[-1]]

    recall_weighted = classification_report_df["recall"][classification_report_df.index[-1]]

    f1_score_weighted = classification_report_df["f1-score"][classification_report_df.index[-1]]

    

    print("f1_score_weighted = ", f1_score_weighted)

    print("precision_weighted = ", precision_weighted)

    print("recall_weighted = ", recall_weighted)

    

    

    metrics.loc["f1_score", name] = f1_score_weighted

    metrics.loc['precision_score', name] = precision_weighted

    metrics.loc['recall_score', name] = recall_weighted

    

    metrics.loc['accuracy', name] = accuracy

    
dataframe_clacc_0_and_1 =  data.iloc[0:60,1:15]

dataframe_licht.append(dataframe_clacc_0_and_1)



dataframe_clacc_0_and_1.head()
dataframe_clacc_2_and_3 =  data.iloc[60:120,1:15]

dataframe_licht.append(dataframe_clacc_2_and_3)

dataframe_clacc_2_and_3.head()
dataframe_clacc_4_and_5 =  data.iloc[120:180,1:15]

dataframe_licht.append(dataframe_clacc_4_and_5)

dataframe_clacc_4_and_5.head()
dataframe_clacc_6_and_7 =  data.iloc[180:240,1:15]

dataframe_licht.append(dataframe_clacc_6_and_7)

dataframe_clacc_6_and_7.head()
gh = 0

heatmap_correlation_licht = []



for i in dataframe_licht:

    print("-----------------------" + class_name[gh] + "------------------------------------")

    corr_data = i[i.columns[0:]].corr()['Class'][:-1].sort_values(ascending=False)

    corr_heatmap = i.corr()

    heatmap_correlation_licht.append(corr_heatmap)       

    hg = 0

    df_index_licht = []

    for hg in range(5):

        df_index_licht.append(corr_data.index[hg])

    feature = i[df_index_licht]

    target = i[['Class']]

    print(feature)

    print(target)

    predict_uith_metric(class_name[gh], feature, target)

    gh = gh + 1

    
metrics_in_percentage = 100*metrics 

metrics_in_percentage
gh = 0

for heatmap_correlation_data in heatmap_correlation_licht: 

    print("\n\n-----|||||||||     heatmap chart of correlation data of "+class_name[gh]+"     ||||||||-----\n\n")

    rcParams['figure.figsize'] = 16, 10

    ax = sns.heatmap(heatmap_correlation_data,xticklabels=heatmap_correlation_data.columns,yticklabels=heatmap_correlation_data.columns)

    plt.savefig("heatmap correlation data chart of " + class_name[gh], dpi=600)

    gh = gh + 1

    plt.show() 

    # break

    

   
metrics_in_percentage = (100*metrics)

font = font_manager.FontProperties(family='Lucida Fax', size=100)

rcParams['font.family'] = 'Britannic Bold'

fig, ax = plt.subplots(figsize = (73, 42)) 

metrics_in_percentage.plot(kind = 'barh', ax = ax, fontsize = 90) 

#plt.rcParams["font.family"] = "Monotype Corsiva"



title_font = {'fontname':'Monotype Corsiva'}

#legend_font = {'fontname':'Impact'}



plt.title(label = "output details of classification on dataset", color = "green", fontsize = 300, loc = "center", fontweight = "bold", **title_font)

legend = ax.legend()





legend = ax.legend(loc = "upper right", labelspacing=2, borderpad=0.45, prop=font)

legend

frame = legend.get_frame()

frame.set_facecolor("#EBF1DE")

frame.set_edgecolor('chartreuse')

frame.set_linewidth(10)

ax.margins(0.6)

ax.grid()

plt.savefig("output details of classification on dataset")