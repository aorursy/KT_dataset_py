# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def univ_scatter(df, features, yname, n=4, writefolder=None):



  for feature in features:



    # tirando os nans

    df_temp = df[~np.isnan(df[feature])][[feature, yname]]



    bins_pos = np.percentile(df_temp[feature].values, np.linspace(0,100,n+1))

    v_mean = list()

    v_std = list()

    

    if bins_pos.size == np.unique(bins_pos).size: # variavel continua

      hist, _ = np.histogram(df_temp[feature], bins_pos)

      xtickslabel = list()

      bin_pos_label = list()

      for i in range(bins_pos.size-1): # vou pegar cada intervalo agora e calcular a media de y

        v = df_temp[(df_temp[feature].values >= bins_pos[i]) & (df_temp[feature].values < bins_pos[i+1])][yname].values

        if np.isnan(v.mean()) or np.isnan(v.std()) or abs(v.mean())==float('inf') or abs(v.std())==float('inf'):

          continue

        else:

          xtickslabel.append('['+str('%.3f'%bins_pos[i])+'-'+str('%.3f'%bins_pos[i+1])+'[')

          v_mean.append(v.mean())

          v_std.append(v.std())

          bin_pos_label.append((bins_pos[i]+bins_pos[i+1])/2)



      v_mean = np.array(v_mean)

      v_std = np.array(v_std)/np.sqrt(hist)

      

      fig, ax1 = plt.subplots()

      ax1.set_xlabel(feature)

      ax1.set_ylabel('mean ' + yname)

      ax1.set_ylim([0, (v_mean+v_std).max()*1.05])

      ax1.set_xticks(bin_pos_label)

      #ax1.plot(bins_pos[:-1], v_mean, label='mean '+yname)

      ax1.plot(bin_pos_label, v_mean, 'o-', label='mean '+yname)

      ax1.set_xticklabels(xtickslabel, rotation=35)

      #ax1.fill_between(bins_pos[:-1], v_mean + v_std, v_mean - v_std, alpha=0.1, color='b')

      ax1.fill_between(bin_pos_label, v_mean + v_std, v_mean - v_std, alpha=0.1, color='b')

      

      color = 'tab:red'

      ax2 = ax1.twinx()

      #ax2.plot(bins_pos[:-1], hist, 'o-', label='bin count', color=color)

      ax2.plot(bin_pos_label, hist, 'o--', label='bin count', color=color)

      ax2.set_ylim([0, hist.max()*1.2])

      ax2.set_ylabel('bin_count', color=color)

      

      if writefolder:

        feature_ = feature.replace(' ', '_')

        feature_ = feature_.replace('/', '_')

        plt.savefig(writefolder+'/scatter_'+feature_+'.png')

      else:      

        plt.tight_layout()

        plt.show()

    else: # variavel categorica

      bins_pos = np.unique(bins_pos)

      hist = list()

      for value in bins_pos:

        hist.append((df_temp[feature].values==value).sum())

      #hist, _ = np.histogram(df[feature], bins_pos)

      for i in range(bins_pos.size): # vou pegar cada intervalo agora e calcular a media de y

        v = df_temp[df_temp[feature].values == bins_pos[i]][yname].values

        v_mean.append(v.mean())

        v_std.append(v.std())



      v_mean = np.array(v_mean)

      v_std = np.array(v_std)/np.sqrt(hist)

      

      fig, ax1 = plt.subplots()

      ax1.set_xlabel(feature)

      ax1.set_ylabel('mean '+yname)

      ax1.set_ylim([0,(v_mean+v_std).max()*1.05])

      ax1.set_xticks(bins_pos)

      ax1.plot(bins_pos, v_mean, 'o-', label='mean '+yname)

      ax1.fill_between(bins_pos, v_mean + v_std, v_mean - v_std, alpha=0.1, color='b')

      

      color = 'tab:red'

      ax2 = ax1.twinx()

      ax2.plot(bins_pos, hist, 'o--', label='bin count', color=color)

      ax2.set_ylim([0, np.array(hist).max()*1.2])

      ax2.set_ylabel('bin_count', color=color)

      

      if writefolder:

        plt.savefig(writefolder+'/scatter_'+feature+'.png')

      else:      

        plt.show()

        

input_file = '/kaggle/input/transformed-covid19-dataset/dataset.xlsx'

df = pd.read_excel(input_file)



# column to model

y_name = 'SARS-Cov-2 exam result'



print (df.columns)

features_to_remove = [

    'Patient ID',

    'Patient addmited to regular ward (1=yes. 0=no)',

    'Patient addmited to semi-intensive unit (1=yes. 0=no)',

    'Patient addmited to intensive care unit (1=yes. 0=no)',

    'Urine - Aspect', # some categorical features which I dont want to deal right now

    'Urine - Leukocytes',

    'Urine - Crystals',

    'Urine - Color',

    'Relationship (Patient/Normal)',

    'Urine - Red blood cells', # this variable is .9 or 1 correlated to another (there is some 'inf' here)

    'Vitamin B12', # needs proper cleaning

    'Base excess (arterial blood gas analysis)', # needs proper cleaning

    'Arteiral Fio2', # needs proper cleaning

]





lst_features = list(df.columns)

lst_features.remove(y_name)



for feature in features_to_remove:

    print ({feature})

    lst_features.remove(feature)

# Let's se some insightful figures



feature = 'Rhinovirus_Enterovirus' #good

# This feature have a good amount of records in 0 and 1 cases and there is a clear correlation, records with

# 'Rhinovirus_Enterovirus'==0 have an average of ~.10 frequency of covid and patients with 'Rhinovirus_Enterovirus'==1

# have an average of 0.02.

univ_scatter(df, [feature], y_name, n=4, writefolder=None)



feature = 'Potassium' #bad

# This is a bad feature, there is little to none information. The difference in the y-axis is little and statistical the same

univ_scatter(df, [feature], y_name, n=4, writefolder=None)



feature = 'Platelets' # good

# Good stuff here

univ_scatter(df, [feature], y_name, n=4, writefolder=None)



feature = 'Parainfluenza 4' #bad

# bad stuff here :/, there is almost none records with 'Parainfluenza 4'==1, we cannot trust that -

# - ONE EYE IN THE DATA AND THE ANOTHER ONE IN THE BIN COUNT :) 



univ_scatter(df, [feature], y_name, n=4, writefolder=None)

# Now we are gonna see all the figures, take your time to decide what are the features you see relevant

# Usually this step is done with a specialized professional, machine learning is not for the hacker geeks only folks! ;)

# I'll put my rushed selection later, you can grab there



univ_scatter(df, lst_features, y_name, n=4, writefolder=None)
# This is my feature selection using the previous figures

# You can generate a 1st seletion and a more rigorous 2st selection to compare

# You must do other techniques like RFE, RF feature importances too! This is an initial approach.



features_sel1 = [

  ##'Patient ID',

  'Patient age quantile',

  'SARS-Cov-2 exam result',

  ##'Patient addmited to regular ward (1=yes. 0=no)',

  ##'Patient addmited to semi-intensive unit (1=yes. 0=no)',

  ##'Patient addmited to intensive care unit (1=yes. 0=no)',

  'Hematocrit', #correlacao 1 com hemoglob

  'Hemoglobin', #good

  'Platelets', # good

  'Mean platelet volume ', # good

  #'Red blood Cells',

  ##'Lymphocytes',

  ##'Mean corpuscular hemoglobin concentration (MCHC)',

  'Leukocytes', # amazing!

  'Basophils',

  ##'Mean corpuscular hemoglobin (MCH)',

  'Eosinophils', #good

  ##'Mean corpuscular volume (MCV)', corr 0.9 com MCH

  'Monocytes', #good

  #'Red blood cell distribution width (RDW)',

  ##'Serum Glucose',

  ##'Respiratory Syncytial Virus',

  ##'Influenza A',

  ##'Influenza B',

  ##'Parainfluenza 1',

  ##'CoronavirusNL63',

  'Rhinovirus_Enterovirus',

  ##'Coronavirus HKU1',

  ##'Parainfluenza 3',

  #'Chlamydophila pneumoniae',

  #'Adenovirus',

  #'Parainfluenza 4',

  ##'Coronavirus229E',

  ##'CoronavirusOC43',

  ##'Inf A H1N1 2009',

  ##'Bordetella pertussis',

  ###'Metapneumovirus',

  ##'Neutrophils',

  ##'Urea',

  'Proteina C reativa mg_dL', #good

  'Creatinine', # good

  ##'Potassium',

  ##'Sodium',

  #'Influenza B. rapid test',

  #'Influenza A. rapid test',

  ##'Alanine transaminase',

  'Aspartate transaminase',

  'Gamma-glutamyltransferase ', #good

  ##'Total Bilirubin',

  ##'Direct Bilirubin',

  ##'Indirect Bilirubin',

  ##'Alkaline phosphatase',

  ##'Ionized calcium ',

  ##'Strepto A',

  ##'Magnesium',

  ##'pCO2 (venous blood gas analysis)',

  ##'Hb saturation (venous blood gas analysis)',

  ##'Base excess (venous blood gas analysis)',

  #'pO2 (venous blood gas analysis)',

  ##'Total CO2 (venous blood gas analysis)',

  #'pH (venous blood gas analysis)',

  ##'HCO3 (venous blood gas analysis)',

  #'Rods #',

  ##'Segmented',

  #'Promyelocytes',

  #'Metamyelocytes',

  #'Myelocytes',

  ##'Urine - Aspect',

  ##'Urine - pH',

  ##'Urine - Hemoglobin',

  ##'Urine - Density',

  ##'Urine - Leukocytes',

  ##'Urine - Crystals',

  ##'Urine - Red blood cells',

  ##'Urine - Color',

  ##'Relationship (Patient/Normal)',

  ##'International normalized ratio (INR)',

  ##'Lactic Dehydrogenase',

  ##'Vitamin B12',

  ##'Creatine phosphokinase (CPK) ',

  ##'Ferritin',

  ##'Arterial Lactic Acid',

  ##'Lipase dosage',

  ##'Albumin',

  ##'Hb saturation (arterial blood gases)',

  ##'pCO2 (arterial blood gas analysis)',

  ##'Base excess (arterial blood gas analysis)',

  ##'pH (arterial blood gas analysis)',

  ##'Total CO2 (arterial blood gas analysis)',

  ##'HCO3 (arterial blood gas analysis)',

  ##'pO2 (arterial blood gas analysis)',

  ##'Arteiral Fio2',

  ##'Phosphor',

  ##'ctO2 (arterial blood gas analysis)',

]