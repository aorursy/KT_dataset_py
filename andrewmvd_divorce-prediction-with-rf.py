import shap

import pandas as pd



import seaborn as sns

import plotly.express as px

import matplotlib.pyplot as plt





from tqdm.notebook import tqdm



from IPython.display import YouTubeVideo



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RepeatedKFold, train_test_split

from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score
raw_data  = pd.read_csv('../input/divorce-prediction/divorce_data.csv', delimiter=';')

reference = pd.read_csv('../input/divorce-prediction/reference.tsv', delimiter='|')
def get_reference(i, verbose = True):

    question = reference.loc[i,'description']

    if verbose:

        print('Q{}: {}'.format(i,question))

    else:

        return(question)
# Check missingness

raw_data.isnull().any().sum()
# Plot correlogram

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

plt.figure(figsize=(20,16))

sns.heatmap(raw_data.corr(), cmap='viridis')

plt.show()
def evaluate_model(preds, test_y, verbose = True, threshold = 0.5):

    

    preds_int = preds >= threshold

    

    

    accuracy = accuracy_score(test_y, preds_int)

    roc_auc  = roc_auc_score(test_y, preds)

    pr_auc   = average_precision_score(test_y, preds)

    f1_val   = f1_score(test_y, preds_int)

    

    if verbose:

        print('Accuracy: {}'.format(accuracy))

        print('AUROC:    {}'.format(roc_auc))

        print('PRAUC:    {}'.format(pr_auc))

        print('F1 Score: {}'.format(f1_val))

    

    

    results = [accuracy, roc_auc, pr_auc, f1_val]

    return(results)
def run_experiment(dataframe, n = 100, use_tqdm=True):

    results = pd.DataFrame()

    

    if use_tqdm:

        iterator = tqdm(range(n))

    else:

        iterator = range(n)

    

    for i in iterator:

        train_x, test_x = train_test_split(dataframe,

                                   test_size = 0.3,

                                   random_state = i

                                  )

        

        

        train_y = train_x.pop('Divorce')

        test_y  = test_x.pop('Divorce')

        

        rf_model = RandomForestClassifier()

        rf_model.fit(train_x, train_y)

        

        preds    = rf_model.predict_proba(test_x)[:,1]

        

        current_results = evaluate_model(preds, test_y, verbose = False)

        results = results.append([current_results])

    

    

    results.reset_index(drop=True, inplace=True)

    results.columns = ['accuracy','roc_auc','pr_auc','f1_score']

    return(results, rf_model)
rf_results, rf_model = run_experiment(raw_data)
print('......................................')

print('Experiment results (mean of 100 runs)')

print('......................................')

print('Accuracy: {}'.format(rf_results.accuracy.mean()))

print('AUROC:    {}'.format(rf_results.roc_auc.mean()))

print('PRAUC:    {}'.format(rf_results.pr_auc.mean()))

print('F1 Score: {}'.format(rf_results.f1_score.mean()))
fig = px.box(rf_results.melt(var_name='metric'),

               x = 'metric',

               y = 'value',

               color_discrete_sequence= ['#fc0362'],

               title = 'Distribution of Metric Values Across 100 Runs',

               template = 'plotly_dark'

              )





fig.update_xaxes(title='Metric', gridcolor = 'rgba(240,240,240, 0.05)')

fig.update_yaxes(title='Value', gridcolor = 'rgba(240,240,240, 0.05)')



fig.update_layout({'plot_bgcolor': 'rgba(40, 40, 40, 1.0)',

                   'paper_bgcolor': 'rgba(30, 30, 30, 1.0)',

                  })

fig.show()
# Herzberg two factor theory of motivation

YouTubeVideo('f-qbGAvR4EU', width=800, height=450)
# Gottham four horsemen of apocalipse

YouTubeVideo('1o30Ps-_8is', width=800, height=450)
explainer   = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(raw_data)
# Average feature contribution

plt.title('Average Feature Contribution for each Class')

shap.summary_plot(shap_values, raw_data, plot_type="bar", plot_size = (15,10))



# Keep top 20 most important feature indexes for later

question_list = [17,19,5,20,18,40,39,9,25,11,15,27,38,26,4,3,41,36,14,22]
# Granular feature contribution plot

plt.title('Feature Contribution According to Value')

shap.summary_plot(shap_values[0], raw_data, plot_size = (15,10))
# Question outline of 20 most important features

for question_index in question_list:

    get_reference(question_index)
shap.dependence_plot("Q5", shap_values[1], raw_data, show=False)

plt.title('Dependance Plot: Q5 and Q36')

plt.show()



get_reference(5)

get_reference(36)
for i in range(1,55):

    shap.dependence_plot("Q{}".format(i), shap_values[1], raw_data)