import pandas as pd

import numpy as np



results = pd.DataFrame(columns = ['Tactic', 'Tactic name', 'Model', 'Description', 'Result', 'Score'])
results = results.append({

    'Tactic': 0,

    'Tactic name': 'Baseline',

    'Model': 'rf',

    'Description': 'Random Forest',

    'Result': 0.6648148148148149,

    'Score': 0.54789

}, ignore_index = True)



results
results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'bg',

    'Description': 'Bagging',

    'Result': 0.784126984126984,

    'Score': 0.73833

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'gb',

    'Description': 'Gradient Boosting',

    'Result': 0.7630291005291,

    'Score': 0.72271

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'lg',

    'Description': 'LightGBM',

    'Result': 0.7979497354497355,

    'Score': 0.77311

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'xg',

    'Description': 'XGBoost',

    'Result': 0.6996031746031746,

    'Score': 0.58578

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'knn',    

    'Description': 'k-Nearest Neighbors',

    'Result': 0.691005291005291,

    'Score': 0.63779

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'rf',    

    'Description': 'Random Forests',

    'Result': 0.6648148148148149,

    'Score': 0.54789

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'lr',

    'Description': 'Logistic Regression',

    'Result': 0.6129629629629629,

    'Score': 0.55308

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'xt',

    'Description': 'eXtra-Trees',

    'Result': 0.6128968253968254,

    'Score': 0.53823

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'lda',

    'Description': 'Linear Discriminant Analysis',

    'Result': 0.6032407407407407,

    'Score': 0.58154

}, ignore_index = True)



results = results.append({

    'Tactic': 1,

    'Tactic name': 'Test classifiers',

    'Model': 'gnb',

    'Description': 'Gaussian Naive Bayes',

    'Result': 0.5685185185185185,

    'Score': 0.42161

}, ignore_index = True)



results.query('Tactic == 1')
results = results.append({

    'Tactic': 2,

    'Tactic name': 'Stack classifiers',

    'Model': '[lg]',

    'Description': 'Stack of models: LightGBM',

    'Result': 0.7979497354497355,

    'Score': 0.77311

}, ignore_index = True)



results = results.append({

    'Tactic': 2,

    'Tactic name': 'Stack classifiers',

    'Model': '[lg, gb, knn]',

    'Description': 'Stack of models: LightGBM, Gradient Boosting, k-Nearest Neighbors',

    'Result': 0.7979497354497355,

    'Score': 0.77303

}, ignore_index = True)



results = results.append({

    'Tactic': 2,

    'Tactic name': 'Stack classifiers',

    'Model': '[lg, knn]',

    'Description': 'Stack of models: LightGBM, k-Nearest Neighbors',

    'Result': 0.7979497354497355,

    'Score': 0.77311

}, ignore_index = True)



results = results.append({

    'Tactic': 2,

    'Tactic name': 'Stack classifiers',

    'Model': '[lg, lr, xt]',

    'Description': 'Stack of models: LightGBM, Logistic Regression, eXtra-Trees',

    'Result': 0.7978835978835979,

    'Score': 0.77311

}, ignore_index = True)



results = results.append({

    'Tactic': 2,

    'Tactic name': 'Stack classifiers',

    'Model': '[lg, lda, xt]',

    'Description': 'Stack of models: LightGBM, Linear Discriminant Analysis, eXtra-trees',

    'Result': 0.7978835978835979,

    'Score': 0.77311

}, ignore_index = True)



results.query('Tactic == 2')
results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'lg',

    'Description': 'Optimized LightGBM',

    'Result': 0.8067460317460318,

    'Score': 0.78236

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'xt',

    'Description': 'Optimized eXtra-Trees',

    'Result': 0.8049603174603175,

    'Score': 0.78149

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'gb',

    'Description': 'Optimized Gradient Boosting',

    'Result': 0.8013227513227513,

    'Score': 0.77732

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'rf',

    'Description': 'Optimized Random Forest',

    'Result': 0.8007936507936508,

    'Score': 0.77521

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'bg',

    'Description': 'Optimized Bagging',

    'Result': 0.7933201058201058,

    'Score': 0.75800

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'xg',

    'Description': 'Optimized XGBoost',

    'Result': 0.7694444444444445,

    'Score': 0.73749

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'knn',

    'Description': 'Optimized k-Nearest Neighbors',

    'Result': 0.756547619047619,

    'Score': 0.71808

}, ignore_index = True)



results = results.append({

    'Tactic': 3,

    'Tactic name': 'Hyperparameter optimization',

    'Model': 'lr',

    'Description': 'Optimized Logistic Regression',

    'Result': 0.6446428571428572,

    'Score': 0.58974

}, ignore_index = True)



results.query('Tactic == 3')
results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, gb, bg, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, Gradient Boosting, Bagging, k-Nearest Neighbors',

    'Result': 0.8103174603174602,

    'Score': 0.78889

}, ignore_index = True)



results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, gb, rf, bg, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, Gradient Boosting, Randrom Forest, Bagging, k-Nearest Neighbors',

    'Result': 0.8101190476190476,

    'Score': 0.78955

}, ignore_index = True)



results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, gb, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, Gradient Boosting, k-Nearest Neighbors',

    'Result': 0.8098544973544973,

    'Score': 0.78940

}, ignore_index = True)



results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, k-Nearest Neighbors',

    'Result': 0.8097883597883597,

    'Score': 0.78563

}, ignore_index = True)



results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, bg, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, Bagging, k-Nearest Neighbors',

    'Result': 0.8097222222222222,

    'Score': 0.78382

}, ignore_index = True)



results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, rf, bg, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, Random Forest, Bagging, k-Nearest Neighbors',

    'Result': 0.8090608465608465,

    'Score': 0.78994

}, ignore_index = True)



results = results.append({

    'Tactic': 4,

    'Tactic name': 'Stack optimized classifiers',

    'Model': '[lg, xt, rf, knn]',

    'Description': 'Stack of models: LightGBM, eXtra-Trees, Random Forest, k-Nearest Neighbors',

    'Result': 0.8089285714285714,

    'Score': 0.79018

}, ignore_index = True)



results.query('Tactic == 4')
results = results.append({

    'Tactic': 5,

    'Tactic name': 'Class weight',

    'Model': 'xt',

    'Description': 'Weighted eXtra-Trees',

    'Result': 0.803042328042328,

    'Score': 0.78081

}, ignore_index = True)



results = results.append({

    'Tactic': 5,

    'Tactic name': 'Class weight',

    'Model': 'rf',

    'Description': 'Weighted Random Forest',

    'Result': 0.7955687830687831,

    'Score': 0.76507

}, ignore_index = True)



results = results.append({

    'Tactic': 5,

    'Tactic name': 'Class weight',

    'Model': 'lg',

    'Description': 'Weighted LightGBM',

    'Result': 0.6341269841269841,

    'Score': 0.53320

}, ignore_index = True)



results = results.append({

    'Tactic': 5,

    'Tactic name': 'Class weight',

    'Model': 'lr',

    'Description': 'Weighted Logistic Regression',

    'Result': 0.5964285714285714,

    'Score': 0.31908

}, ignore_index = True)



results.query('Tactic == 5')
results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'xt',

    'Description': 'eXtra-Trees',

    'Result': 0.8222883597883598,

    'Score': 0.80142

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'lg',

    'Description': 'LightGBM',

    'Result': 0.8196428571428571,

    'Score': 0.79496

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'gb',

    'Description': 'Gradient Boosting',

    'Result': 0.8175925925925925,

    'Score': 0.79778

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'rf',

    'Description': 'Random Forests',

    'Result': 0.817063492063492,

    'Score': 0.79113

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'xg',

    'Description': 'XGBoost',

    'Result': 0.8039021164021164,

    'Score': 0.78067

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'bg',

    'Description': 'Bagging',

    'Result': 0.8037037037037037,

    'Score': 0.77487

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'knn',

    'Description': 'k-Nearest Neighbors',

    'Result': 0.7116402116402116,

    'Score': 0.65645

}, ignore_index = True)



results = results.append({

    'Tactic': 6,

    'Tactic name': 'Feature engineering',

    'Model': 'lr',

    'Description': 'Logistic Regression',

    'Result': 0.6223544973544973,

    'Score': 0.56301

}, ignore_index = True)



# results = results.append({

#     'Tactic': 6,

#     'Tactic name': 'Feature engineering',

#     'Model': 'lda',

#     'Description': 'Linear Discriminant Analysis',

#     'Result': 0.5973544973544973,

#     'Score': 0.56452

# }, ignore_index = True)



# results = results.append({

#     'Tactic': 6,

#     'Tactic name': 'Feature engineering',

#     'Model': 'gnb',

#     'Description': 'Gaussian Naive Bayes',

#     'Result': 0.5212962962962963,

#     'Score': 0.42493

# }, ignore_index = True)



# results = results.append({

#     'Tactic': 6,

#     'Tactic name': 'Feature engineering',

#     'Model': 'ab',

#     'Description': 'AdaBoost',

#     'Result': 0.41064814814814815,

#     'Score': 0.15887

# }, ignore_index = True)



results.query('Tactic == 6')
results = results.append({

    'Tactic': 7,

    'Tactic name': 'Outlier detection',

    'Model': 'lg',

    'Description': 'LightGBM. Contamination 0.01',

    'Result': 0.8020443613041155,

    'Score': 0.78039

}, ignore_index = True)



results = results.append({

    'Tactic': 7,

    'Tactic name': 'Outlier detection',

    'Model': 'lg',

    'Description': 'LightGBM. Contamination 0.05',

    'Result': 0.8041631857421331,

    'Score': 0.77526

}, ignore_index = True)



results = results.append({

    'Tactic': 7,

    'Tactic name': 'Outlier detection',

    'Model': 'lg',

    'Description': 'LightGBM. Contamination 0.1',

    'Result': 0.8026161081716637,

    'Score': 0.76304

}, ignore_index = True)



results = results.append({

    'Tactic': 7,

    'Tactic name': 'Outlier detection',

    'Model': 'lg',

    'Description': 'LightGBM. Contamination 0.2',

    'Result': 0.7978670634920635,

    'Score': 0.74398

}, ignore_index = True)



results.query('Tactic == 7')
results = results.append({

    'Tactic': 8,

    'Tactic name': 'Feature selection: optimal importance',

    'Model': 'lg',

    'Description': 'LightGBM',

    'Result': 0.8232142857142858,

    'Score': 0.79681

}, ignore_index = True)



results = results.append({

    'Tactic': 8,

    'Tactic name': 'Feature selection: optimal importance',

    'Model': 'lg',

    'Description': 'LightGBM. Only positive',

    'Result': 0.8163,

    'Score': 0.79507

}, ignore_index = True)



results.query('Tactic == 8')
results = results.append({

    'Tactic': 9,

    'Tactic name': 'Feature selection: backward elimination',

    'Model': 'lg',

    'Description': 'LightGBM',

    'Result': 0.7986111111111112,

    'Score': 0.79529

}, ignore_index = True)



results.query('Tactic == 9')
results = results.append({

    'Tactic': 10,

    'Tactic name': 'Feature selection: forward selection',

    'Model': 'lg',

    'Description': 'LightGBM',

    'Result': 0.8046957671957673,

    'Score': 0.79155

}, ignore_index = True)



results.query('Tactic == 10')
## Export Tactic00 + Tactic01 results ordered by score

tactic_01_results = results.query('Tactic == 1 or Tactic == 0').sort_values('Score', ascending=False).reset_index(drop=True)

tactic_01_results.to_csv('tactic_01_results.csv', index=True, index_label='Id')



## Export Tactic03 results ordered by score

tactic_03_results = results.query('Tactic == 3').sort_values('Score', ascending=False).reset_index(drop=True)

tactic_03_results.to_csv('tactic_03_results.csv', index=True, index_label='Id')



## Calculate ratio

results['Ratio'] = round(results['Score'] / results['Result'] * 100, 2)



results.to_csv('results.csv', index=True, index_label='Id')

results