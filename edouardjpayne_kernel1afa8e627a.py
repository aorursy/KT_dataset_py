import math

import numpy as np

import pandas as pd

import datetime as dt

import scipy.stats as sp

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import RBF
df = pd.read_csv("../input/WaveformSummary.csv").set_index("subj.num")
df["var.signal"] = df["sd.signal"] * df["sd.signal"] 
#This code below plots the charts individually and saves the data to be accessed later



subject_number = [8,12,19]

kernels = []

gaussian_regression = []

sampled_data = []

colours = ["blue","orange","green"]

chart_data = []









graphs = []

for i,x in enumerate(subject_number):

    plt.figure(num=None, figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k')

    plt.ylim((-2, 3)) 

    plt.xlabel("Time since peak (sec)") 

    plt.ylabel("Standardised signal")

    plt.title("GP regression model for each Subject Number "+str(x))

    label_text = "Subject Number " + str(subject_number[i])

    



    initial_ell = 0.1  #Create a range that is 0.1 part

    initial_A = np.std(df.loc[x]['avg.signal']) # SD of the mean of the training data

    initial_scale = initial_A * initial_A # Create the variance of the data



    

    x_outout = df.loc[x]["secs.since.peak"].values.reshape(-1,1) #Get the values you want to retrieve



    y = df.loc[x]["avg.signal"] #Get the values you want to predict

    



    rbf_kernel = initial_scale * RBF(length_scale=initial_ell, 

    length_scale_bounds=(initial_ell/40, 5 * initial_ell) ) #Create the boundaries and kernel

    

    gp = GaussianProcessRegressor(kernel=rbf_kernel, alpha=df.loc[x]["var.signal"]) #Using the Kernel and variance create GP

    

    gp.fit(x_outout,y) #Fit the values

    

    

    ordinal_dates = np.linspace(0,11,12)/10 #Specify the points we want to sample

    date_col_vec = ordinal_dates[:,np.newaxis] #Transpose these samples

    

    nSamples = 1000 #Specify how many samples we want to generate

    

    values = gp.sample_y(date_col_vec, nSamples ) # Generate the samples 

    

    sampled_data.append(values) # Append the values to an array (this will be used for the score)

    

    y_mean, y_std = gp.predict(date_col_vec, return_std=True) # Create y_mean and y_std, this will be used for the chart 

    

    chart_data.append([y_mean,y_std,x_outout, y]) #Save this data so that we can use it later when plotting charts 

    #together

    

    

    plt.plot(ordinal_dates,y_mean,lw=2, label=label_text,color=colours[i]) # Create a line of the predicted values

    plt.fill_between(ordinal_dates, y_mean - y_std,y_mean+y_std, alpha=0.2,color=colours[i]) #Plot the range



    plt.scatter(x_outout,y,color=colours[i], alpha=0.5) #Plot the data from the dataframe

    

    plt.show()

    



    
#This code below plots the charts overlapped on top of each other



i = 0

plt.figure(num=None, figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k')

for model in chart_data:

    ordinal_dates = np.linspace(0,11,12)/10

    plt.ylim((-2, 3.5)) 

    plt.xlabel("Time since peak (sec)")

    plt.ylabel("Standardised signal")

    plt.title("GP regression model for each Subject Number")

    label_text = "Subject Number " + str(subject_number[i])

    plt.plot(ordinal_dates,model[0],lw=2, label=label_text) # Create a line of the predicted values

    plt.fill_between(ordinal_dates, model[0] - model[1],model[0]+model[1], alpha=0.2)  #Plot the range



    plt.scatter(model[2],model[3],color=colours[i], alpha=0.5) #Plot the data from the dataframe

    i +=1



plt.legend(loc="upper left")



def check_min(value):

    #Returns the minimum value of the traces

    return np.min(value, axis=0)



def check_max(value):

    #Returns the maximum value of the traces

    return np.max(value, axis=0)



def check_mean(value):

    #Returns the mean values of the traces

    return np.mean(value, axis=0)
def create_model(function, data, bins):

    returned_data = [] #Array which we will save the data once we have applied the function

    plt.figure(num=None, figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k') #Initalize the figure size

    label_names = ["Subject Number 8", "Subject Number 12", "Subject Number 19"] #Set the subject numbers

    plt.xlabel("Scores") # Plot the score labels

    plt.ylabel("Distribution") # Plot the distibution labels

    plt.title("Distribution of scores for "+function.__name__+" function") # Plot the title of the distibution

    for i in data:

        returned_data.append(function(i)) #Append the returned data into an array



    increment = 0 #Set a variable that will be used to reference labels

    

    for i in returned_data:

        x = sns.distplot(i, bins = bins, label=label_names[increment]); #Create histogramsfor each returned data

        increment +=1

    plt.legend()  

    



    return returned_data





def analyse_score(data):

    min_values = math.floor(np.min(data)) # This will return the minimum value of the score

    max_values = math.ceil(np.max(data))  # This will return the maximum value of the scores

    bins_to_use = np.arange(min_values,max_values,0.02) # This is the number of bins we will use

    model_bins = {}



    #Each model will have the same number of bins that are qually spaced

    model_a_values, model_a_bins, _ = plt.hist(data[0], bins=bins_to_use) 

    model_b_values, model_b_bins, _ = plt.hist(data[1], bins=bins_to_use)

    model_c_values, model_c_bins, _ = plt.hist(data[2], bins=bins_to_use)

    plt.close()



    #Create a dictionary

    model_bins['model_a'] = {}

    model_bins['model_b'] = {}

    model_bins['model_c'] = {}



    model_bins['model_a']['values'] = model_a_values

    model_bins['model_a']['bins'] = bins_to_use

    model_bins['model_b']['values'] = model_b_values

    model_bins['model_a']['bins'] = bins_to_use

    model_bins['model_c']['values'] = model_c_values

    model_bins['model_a']['bins'] = bins_to_use

    

    #Append the results to numpy array

    len_bins = len(bins_to_use)-1

    list_value = []

    for i in range(len_bins-1):

        values = []

        for key in model_bins:

            values.append(model_bins[key]['values'][i])



        list_value.append(values)



    list_value = np.array(list_value)   

    

    #THIS SECTION CALCULATES SCORE OF ALL SUBJECTS

    

    #Identify the score of each array this is done by summing the row with the highest value using np.amax

    identified_correctly = np.sum(np.amax(list_value, axis=1))

    total = np.sum(list_value)

    

    #The score is calculated by dividing the results that are identified correctly by the total

    score = round((identified_correctly / total),3)

    

    

    subject_number = ["All Subjects","8","12","19"]

    correct_results = [identified_correctly]

    wrong_results = [total-identified_correctly]

    model_results = [total]

    mode_accuracy = [score]

    

    #THIS SECTION CALCULATES SCORE OF EACH SUBJECT

    for i in range(len(data)):

        get_hist_values = list_value[list_value[:,i]>0] #Returns the hist values for each subject



        correct_model_data = get_hist_values[(np.argmax(list_value[list_value[:,i]>0],axis=1))==i] #Returns the values that were correctly identified

        incorrect_model_data = get_hist_values[(np.argmax(list_value[list_value[:,i]>0],axis=1))!=i] #Returns the values that were not correctly identified

        

        a = [i]

        correct_filter = [True if i in a else False for i in range(len(data))] #Use this to filter correct model data





        correct_values = np.sum(correct_model_data[:,correct_filter]) #Filter out incorrect values

        total_values = np.sum(get_hist_values[:,correct_filter])  #Get total values values

        

        mode_accuracy.append(correct_values/total_values)

        correct_results.append(correct_values)

        wrong_results.append(total_values-correct_values)

        model_results.append(total_values)

        

    #This section is to create the dataframe

    d = {'Subject Number': subject_number, 'Correct': correct_results,'Wrong ': wrong_results, 'Total ': model_results,'Accuracy': mode_accuracy}

    df = pd.DataFrame(data=d)

    df['Accuracy'] = df['Accuracy'] * 100

    df['Accuracy'] = df['Accuracy'].astype(str) + '%'

    df.set_index('Subject Number', inplace=True)



    return df

check_min_values = create_model(check_min,sampled_data, 50)
check_max_values = create_model(check_max,sampled_data, 50)
check_mean_values = create_model(check_mean,sampled_data, 50)
analyse_score(check_min_values)
analyse_score(check_max_values)
analyse_score(check_mean_values)