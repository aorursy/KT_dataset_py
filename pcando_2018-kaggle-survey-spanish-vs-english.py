import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager as fm
import time
import warnings
warnings.filterwarnings("ignore")
# Set fonts-style for plots
FONT_PATH = "../input/source-sans/SourceSansPro-Regular.ttf" 
prop = fm.FontProperties(fname=FONT_PATH) # Labels and ticks

FILE_NAME = "multipleChoiceResponses.csv"
DATA_PATH = "../input/kaggle-survey-2018/"
try:
    %time multiple_choice2018 = pd.read_csv(f"{DATA_PATH}{FILE_NAME}", header=1)
    print(f"SUCCESS: {FILE_NAME} in {DATA_PATH} has been successfully read")
    print("         Number of rows in dataset: " + str(multiple_choice2018.shape[0]))
    print("         Number of columns in dataset: " + str(multiple_choice2018.shape[1]))
except:
    print(f"ERROR: Sorry, could not load {FILE_NAME}.")
    
# Creating Spanish speaking dataset
spanish_speaking_countries = ["Spain", "Mexico", "Colombia", "Argentina", "Peru", "Chile"]
kaggle_spanish = multiple_choice2018[multiple_choice2018["In which country do you currently reside?"].isin(spanish_speaking_countries)]

# Creating English speaking dataset
english_speaking_countries = ["United States of America", "United Kingdom of Great Britain and Northern Ireland","Ireland","Canada","Australia", "New Zealand"]
kaggle_english = multiple_choice2018[multiple_choice2018["In which country do you currently reside?"].isin(english_speaking_countries)]
# Create dataset
spanish_response = kaggle_spanish[["Duration (in seconds)", "In which country do you currently reside?"]].groupby("In which country do you currently reside?").sum()
spanish_response["Number of respondants"] = kaggle_spanish[["Duration (in seconds)", "In which country do you currently reside?"]].groupby("In which country do you currently reside?").size()
spanish_response = spanish_response.reset_index()
spanish_response["average response time per country (seconds)"] = np.array(spanish_response["Duration (in seconds)"]) / np.array(spanish_response["Number of respondants"])
spanish_response["average response time per country (mins)"] = spanish_response["average response time per country (seconds)"].apply(lambda x: round(x * 0.016667,0))

english_response = kaggle_english[["Duration (in seconds)", "In which country do you currently reside?"]].groupby("In which country do you currently reside?").sum()
english_response["Number of respondants"] = kaggle_english[["Duration (in seconds)", "In which country do you currently reside?"]].groupby("In which country do you currently reside?").size()
english_response = english_response.reset_index()
english_response["average response time per country (seconds)"] = np.array(english_response["Duration (in seconds)"]) / np.array(english_response["Number of respondants"])
english_response["average response time per country (mins)"] = english_response["average response time per country (seconds)"].apply(lambda x: round(x * 0.016667,0))

# Avg. response dataset
avg_response = spanish_response.append(english_response, ignore_index=True)

# Create figure object
fig, ax = plt.subplots(figsize=(10,8))

# Plot title
fig.text(0.95, 0.001, "Average Survey Response Times by Spanish and English Speaking Countries", fontproperties=prop, fontsize=14, position=(0.122, 0.93))

# Plot line graph of avg. response times
ax.plot(avg_response.index, avg_response["average response time per country (mins)"], color="#B4DBDA")

# Grid line settings
ax.grid(True, color="grey", alpha=0.1, linestyle="-")

# Plot background colour
ax.set_facecolor("white")

# Plot ylabel settings
ax.set_ylabel("Average Response Time (Mins.)", fontproperties=prop, fontsize=13, labelpad=30)

# Plot yticks
plt.yticks(rotation=0, fontproperties=prop)

# Plot xlabel settings
ax.set_xlabel("Country", fontproperties=prop, fontsize=13)

# Plot xticks
countries = avg_response["In which country do you currently reside?"].values
countries[10] = "United Kingdom"
countries[11] = "United States"
countries
plt.xticks(np.arange(12), countries, fontproperties=prop, fontsize=11, rotation=70)

# Plot title 
# ax.set_title("English & Spanish Speaking Country Average Survey Response Time", fontproperties=prop, fontsize=14, pad=20)

# Plot border line settings
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
# Dataset for plot
country = kaggle_spanish["In which country do you currently reside?"].value_counts().to_frame()
country_eng = kaggle_english["In which country do you currently reside?"].value_counts().to_frame()

# Create figure object
fig, ax = plt.subplots(figsize=(10,8))

# Plot title
fig.text(0.95, 0.001, "English & Spanish Speaking Country Respondant Distribution", fontproperties=prop, fontsize=14, position=(0.122, 0.93))

# Plot bar - Spanish speaking countries
ax.bar(spanish_response["In which country do you currently reside?"], country["In which country do you currently reside?"].values,
       alpha=0.6, 
       color='#7FB3D5', 
       linewidth=0.8, 
       edgecolor="black")

# Plot bar - English speaking countries
ax.bar(english_response["In which country do you currently reside?"], country_eng["In which country do you currently reside?"].values,
       alpha=0.6, 
       color='#1F618D', 
       linewidth=0.8, 
       edgecolor="black")

# Grid line settings
ax.grid(True, color="grey", alpha=0.1, linestyle="-")

# Plot background colour
ax.set_facecolor("white")

# Plot ylabel settings
ax.set_ylabel("Number of 2018 Kaggle Survey Respondants", fontproperties=prop, fontsize=13, labelpad=30)

# Plot yticks
plt.yticks(rotation=0, fontproperties=prop)

# Plot xlabel settings
ax.set_xlabel("Country", fontproperties=prop, fontsize=13, labelpad=20)

# Plot xticks
countries = ('Spain','Mexico','Colombia','Argentina','Peru','Chile','United States','United Kingdom','Canada','Australia','Ireland','New Zealand')
plt.xticks(np.arange(12), countries, fontproperties=prop, fontsize=11, rotation=70)

# # Plot title 
# ax.set_title("English & Spanish Speaking Country Respondant Distribution", fontproperties=prop, fontsize=14, pad=20)

# Plot border line settings
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# Adding country values above bar
for i, v in enumerate(np.append(country["In which country do you currently reside?"].values, country_eng["In which country do you currently reside?"].values)):
    ax.text(x=i - .18, y=v + 100, s=str(v), fontproperties=prop, fontsize=12);
# Data
spanish_sex = pd.get_dummies(kaggle_spanish[["What is your gender? - Selected Choice", "In which country do you currently reside?"]]["What is your gender? - Selected Choice"]).join(kaggle_spanish["In which country do you currently reside?"]).groupby('In which country do you currently reside?').sum()
english_sex = pd.get_dummies(kaggle_english[["What is your gender? - Selected Choice", "In which country do you currently reside?"]]["What is your gender? - Selected Choice"]).join(kaggle_english["In which country do you currently reside?"]).groupby('In which country do you currently reside?').sum()

# Create figure object
fig, ax = plt.subplots(figsize=(10,8))

# Plot title
fig.text(0.95, 0.001, "Respondant Gender Distributions for English and Spanish Speaking Countries", fontproperties=prop, fontsize=14, position=(0.122, 0.93))

# Plot bar - Spanish speaking countries
b1 = ax.bar(spanish_sex.index, spanish_sex['Prefer to self-describe'].values, bottom=np.array(spanish_sex.Female.values) + np.array(spanish_sex.Male.values) + np.array(spanish_sex['Prefer not to say'].values), 
            alpha=0.6,
            color='#9EB3C2', 
            linewidth=0.8, 
            edgecolor="black")
b2 = ax.bar(spanish_sex.index, spanish_sex['Prefer not to say'].values, bottom=np.array(spanish_sex.Female.values) + np.array(spanish_sex.Male.values), 
            alpha=0.6,
            color='#1C7293', 
            linewidth=0.8, 
            edgecolor="black")
b3 = ax.bar(spanish_sex.index, spanish_sex.Female.values,
            alpha=0.6,
            color='#065A82', 
            linewidth=0.8, 
            edgecolor="black")
b4 = ax.bar(spanish_sex.index, spanish_sex.Male.values, bottom=spanish_sex.Female.values,
            alpha=0.6,
            color='#1B3B6F', 
            linewidth=0.8, 
            edgecolor="black")
ax.bar(english_sex.index, english_sex['Prefer to self-describe'].values, bottom=np.array(english_sex.Female.values) + np.array(english_sex.Male.values) + np.array(english_sex['Prefer not to say'].values), 
            alpha=0.6,
            color='#9EB3C2', 
            linewidth=0.8, 
            edgecolor="black")
ax.bar(english_sex.index, english_sex['Prefer not to say'].values, bottom=np.array(english_sex.Female.values) + np.array(english_sex.Male.values), 
            alpha=0.6,
            color='#1C7293', 
            linewidth=0.8, 
            edgecolor="black")
ax.bar(english_sex.index, english_sex.Female.values,
            alpha=0.6,
            color='#065A82', 
            linewidth=0.8, 
            edgecolor="black")
ax.bar(english_sex.index, english_sex.Male.values, bottom=english_sex.Female.values,
            alpha=0.6,
            color='#1B3B6F', 
            linewidth=0.8, 
            edgecolor="black")

# Legend
ax.legend((b1[0], b2[0], b3[0], b4[0]), ('Prefer to self-describe', 'Prefer not to say', 'Female', 'Male'), bbox_to_anchor=(0.29, 0.99))

# Grid line settings
ax.grid(True, color="grey", alpha=0.1, linestyle="-")

# Plot ylabel settings
ax.set_ylabel("Number of 2018 Kaggle Survey Respondants", fontproperties=prop, fontsize=13, labelpad=30)

# Plot yticks
plt.yticks(rotation=0, fontproperties=prop)

# Plot xlabel settings
ax.set_xlabel("Country", fontproperties=prop, fontsize=13, labelpad=20)

# Plot xticks
countries = ('Argentina', 'Chile','Colombia','Mexico', 'Peru','Spain','Australia','Canada', 'Ireland','New Zealand', 'United Kingdom', 'United States')
plt.xticks(np.arange(12), countries, fontproperties=prop, fontsize=11, rotation=70)

# Plot border line settings
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
# Create figure object
fig2, ax2 = plt.subplots(1, 2, figsize=(10,8))

# Plot title
fig2.text(0.95, 0.001, "Respondant Gender Distributions for English and Spanish Speaking Countries (Detail)", fontproperties=prop, fontsize=14, position=(0.122, 0.93))

# Plot xlabel
fig2.text(0.95, 0.001, "Country", fontproperties=prop, fontsize=13, position=(0.49, -0.01))

# Amending x-label tick data
countries = english_sex.index.values
countries[3] = "NZ"
countries[4] = "UK"
countries[5]= "US"

# Plot bar - English speaking countries
ax2[0].bar(countries, english_sex['Prefer to self-describe'].values, bottom=np.array(english_sex.Female.values) + np.array(english_sex.Male.values) + np.array(english_sex['Prefer not to say'].values), 
            alpha=0.6,
            color='#9EB3C2', 
            linewidth=0.8, 
            edgecolor="black")
ax2[0].bar(countries, english_sex['Prefer not to say'].values, bottom=np.array(english_sex.Female.values) + np.array(english_sex.Male.values), 
            alpha=0.6,
            color='#1C7293', 
            linewidth=0.8, 
            edgecolor="black")
ax2[0].bar(countries, english_sex.Female.values,
            alpha=0.6,
            color='#065A82', 
            linewidth=0.8, 
            edgecolor="black")
ax2[0].bar(countries, english_sex.Male.values, bottom=english_sex.Female.values,
            alpha=0.6,
            color='#1B3B6F', 
            linewidth=0.8, 
            edgecolor="black")

# Grid line settings
ax2[0].grid(True, color="grey", alpha=0.1, linestyle="-")

# Plot ylabel settings
ax2[0].set_ylabel("Number of 2018 Kaggle Survey Respondants", fontproperties=prop, fontsize=13, labelpad=30)

# Plot yticks
plt.yticks(rotation=0, fontproperties=prop)

# Plot xticks
ax2[0].set_xticklabels(countries, rotation=70, fontproperties=prop, fontsize=11)

# Plot border line settings
ax2[0].spines['top'].set_visible(False)
ax2[0].spines['right'].set_visible(False)
ax2[0].spines['bottom'].set_visible(True)
ax2[0].spines['left'].set_visible(True)

# # Plot bar - Spanish speaking countries
ax2[1].bar(spanish_sex.index, spanish_sex['Prefer to self-describe'].values, bottom=np.array(spanish_sex.Female.values) + np.array(spanish_sex.Male.values) + np.array(spanish_sex['Prefer not to say'].values), 
            alpha=0.6,
            color='#9EB3C2', 
            linewidth=0.8, 
            edgecolor="black")
ax2[1].bar(spanish_sex.index, spanish_sex['Prefer not to say'].values, bottom=np.array(spanish_sex.Female.values) + np.array(spanish_sex.Male.values), 
            alpha=0.6,
            color='#1C7293', 
            linewidth=0.8, 
            edgecolor="black")
ax2[1].bar(spanish_sex.index, spanish_sex.Female.values,
            alpha=0.6,
            color='#065A82', 
            linewidth=0.8, 
            edgecolor="black")
ax2[1].bar(spanish_sex.index, spanish_sex.Male.values, bottom=spanish_sex.Female.values,
            alpha=0.6,
            color='#1B3B6F', 
            linewidth=0.8, 
            edgecolor="black")

# Grid line settings
ax2[1].grid(True, color="grey", alpha=0.1, linestyle="-")

# Plot yticks
plt.yticks(rotation=0, fontproperties=prop)
ax2[1].yaxis.tick_right()

# Plot xticks
ax2[1].set_xticklabels(spanish_sex.index, rotation=70, fontproperties=prop, fontsize=11)

# Plot border line settings
ax2[1].spines['top'].set_visible(False)
ax2[1].spines['right'].set_visible(True)
ax2[1].spines['bottom'].set_visible(True)
ax2[1].spines['left'].set_visible(False)
# Datasets
spanish_jobtitle = kaggle_spanish["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].value_counts()
spanish_jobtitle = spanish_jobtitle.to_frame()
spanish_jobtitle["proportion"] = spanish_jobtitle["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].apply(lambda x:  x / spanish_jobtitle["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].sum() * 100)
spanish_jobtitle = spanish_jobtitle.append(pd.DataFrame([[0, 0]], columns=["Select the title most similar to your current role (or most recent title if retired): - Selected Choice", "proportion"]))
spanish_jobtitle.rename({0:"Data Journalist"}, axis="index", inplace=True)

english_jobtitle = kaggle_english["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].value_counts()
english_jobtitle = english_jobtitle.to_frame()
english_jobtitle["proportion"] = english_jobtitle["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].apply(lambda x:  x / english_jobtitle["Select the title most similar to your current role (or most recent title if retired): - Selected Choice"].sum() * 100)

# Create figure object
fig3, ax3 = plt.subplots(1, 2, figsize=(10,8), sharey=True, sharex=True)
plt.subplots_adjust(wspace=0.01)

# Plot x-label
fig3.text(0.95, 0.001, "Proportion of 2018 Kaggle Survey Respondants (% Percentage)", fontproperties=prop, fontsize=13, ha='center', position=(0.52, 0.03))
                                                                                                                               
# Plot xticks
ax3[0].set_xticklabels(["", "5", "10", "15", "20"], rotation=70, fontproperties=prop, fontsize=11)
ax3[1].set_xticklabels(["", "5", "10", "15", "20"], rotation=70, fontproperties=prop, fontsize=11)

# Plot title
fig3.text(0.95, 0.001, "Spanish and English Speaking Respondant Current Employment Distribution", fontproperties=prop, fontsize=14, ha='center', position=(0.43, 0.91))

# Plot Spanish speaking job categories
ax3[0].barh(spanish_jobtitle.index, 
            spanish_jobtitle.proportion,
            alpha=0.6,
            color="#7FB3D5",
            linewidth=0.8, 
            edgecolor="black")

# Graph y-label
ax3[0].set_yticklabels(english_jobtitle.index, fontproperties=prop, fontsize=12)

# Plot spines
ax3[0].invert_xaxis()
ax3[0].spines['top'].set_visible(False)
ax3[0].spines['right'].set_visible(False)
ax3[0].spines['bottom'].set_visible(True)
ax3[0].spines['left'].set_visible(False)
ax3[1].spines['top'].set_visible(False)
ax3[1].spines['right'].set_visible(False)
ax3[1].spines['bottom'].set_visible(True)
ax3[1].spines['left'].set_visible(False)

# Plot English speaking job categories
ax3[1].barh(english_jobtitle.index, 
            english_jobtitle.proportion,
            alpha=0.6,
            color="#1F618D",
            linewidth=0.8, 
            edgecolor="black")

# Invert axes
ax3[1].invert_yaxis()
ax3[1].invert_xaxis()

# Grid line settings
ax3[0].grid(True, color="grey", alpha=0.1, linestyle="-")

# Grid line settings
ax3[1].grid(True, color="grey", alpha=0.1, linestyle="-")

# Tick parameters
ax3[1].tick_params(axis="y", left=False)
ax3[0].tick_params(axis="y", left=False)
# Dataset
spanish_ed = kaggle_spanish["What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"].value_counts().to_frame()
spanish_ed["Proportion"] = spanish_ed["What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"].apply(lambda x: x / kaggle_spanish.shape[0] * 100)
english_ed = kaggle_english["What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"].value_counts().to_frame()
english_ed["Proportion"] = english_ed["What is the highest level of formal education that you have attained or plan to attain within the next 2 years?"].apply(lambda x: x / kaggle_english.shape[0] * 100)

# Create figure and axes objects
fig4, ax4 = plt.subplots(1, 2, figsize=(10,8), sharey=True, sharex=True)

# Create spanish_label array for x-labels
spanish_labels = spanish_ed.index.values
spanish_labels[4] = "Study with No Degree"
spanish_labels[5] = "High School"
spanish_labels[6] = "No Response"

# Plot Spanish speaking education data
ax4[0].barh(spanish_ed.index, spanish_ed.Proportion, 
            alpha=0.6,
            color="#7FB3D5",
            linewidth=0.8,
            edgecolor="black")

# Plot English speaking education data
ax4[1].barh(english_ed.index, english_ed.Proportion, alpha=0.6,
            color="#1F618D",
            linewidth=0.8, 
            edgecolor="black")

# Plot x-label
fig4.text(0.95, 0.001, "Proportion of 2018 Kaggle Survey Respondants (% Percentage)", fontproperties=prop, fontsize=13, ha='center', position=(0.52, 0.03))

# Plot title
fig4.text(0.95, 0.001, "Spanish and English Speaking Respondant Higher Education Distribution", fontproperties=prop, fontsize=14, ha='center', position=(0.415, 0.95))

# Grid line settings
ax4[0].grid(True, color="grey", alpha=0.1, linestyle="-")

# Grid line settings
ax4[1].grid(True, color="grey", alpha=0.1, linestyle="-")

# Plot spines
ax4[0].invert_xaxis()
ax4[0].spines['top'].set_visible(False)
ax4[0].spines['right'].set_visible(False)
ax4[0].spines['bottom'].set_visible(True)
ax4[0].spines['left'].set_visible(True)
ax4[1].spines['top'].set_visible(False)
ax4[1].spines['right'].set_visible(False)
ax4[1].spines['bottom'].set_visible(True)
ax4[1].spines['left'].set_visible(True)

# Invert axes
ax4[1].invert_yaxis()
ax4[1].invert_xaxis()

# Tick parameters
ax4[1].tick_params(axis="y", left=False)
ax4[0].tick_params(axis="y", left=False)

# Graph y-label
ax4[0].set_yticklabels(spanish_labels, fontproperties=prop, fontsize=12)

ax4[0].set_title("Spanish", fontproperties=prop, fontsize=13)
ax4[1].set_title("English", fontproperties=prop, fontsize=13);
# Dataset
category = ["I do not wish to disclose my approximate yearly compensation",
           "0-10,000",
           "10-20,000",
           "20-30,000",
           "30-40,000",
           "40-50,000",
           "50-60,000",
           "60-70,000",
           "70-80,000",
           "80-90,000",
           "90-100,000",
           "100-125,000",
           "125-150,000",
           "150-200,000",
           "200-250,000",
           "250-300,000",
           "300-400,000",
           "400-500,000",
           "500,000+"]

spanish_salary = kaggle_spanish["What is your current yearly compensation (approximate $USD)?"].value_counts().to_frame()
spanish_salary["percentage"] = spanish_salary["What is your current yearly compensation (approximate $USD)?"].apply(lambda x: x / spanish_salary["What is your current yearly compensation (approximate $USD)?"].sum() * 100)
spanish_salary = spanish_salary.reset_index().rename(columns={"index":"yearly_compensation", "What is your current yearly compensation (approximate $USD)?":"count"})
further_categories = pd.DataFrame([["400-500,000", 0, 0.0], ["250-300,000", 0, 0.0]], columns=["yearly_compensation", "count", "percentage"])
spanish_salary = pd.concat([further_categories, spanish_salary], axis=0)
spanish_salary["yearly_compensation"] = pd.Categorical(spanish_salary["yearly_compensation"], ordered=True, categories=category)
spanish_salary = spanish_salary.sort_values(by="yearly_compensation")

english_salary = kaggle_english["What is your current yearly compensation (approximate $USD)?"].value_counts().to_frame()
english_salary["percentage"] = english_salary["What is your current yearly compensation (approximate $USD)?"].apply(lambda x: x / english_salary["What is your current yearly compensation (approximate $USD)?"].sum() * 100)
english_salary = english_salary.reset_index().rename(columns={"index":"yearly_compensation", "What is your current yearly compensation (approximate $USD)?":"count"})
english_salary["yearly_compensation"] = pd.Categorical(english_salary["yearly_compensation"], ordered=True, categories=category)
english_salary = english_salary.sort_values(by="yearly_compensation")

salary_labels = list(spanish_salary.yearly_compensation.values)
salary_labels[0] = "Not Disclosed"
salary_labels

# Create figure and axes objects
fig5, ax5 = plt.subplots(1, 2, figsize=(12,8), sharey=True)

# Plot Spanish speaking salary data
ax5[0].scatter(x=np.arange(19), y=spanish_salary["count"], s=spanish_salary.percentage * 40, color="#7FB3D5", alpha=0.6,
            linewidth=0.5,
            edgecolor="black")

# Plot English speaking salary data
ax5[1].scatter(x=np.arange(19), y=english_salary["count"], s=english_salary.percentage * 40, color="#1F618D", alpha=0.6,
            linewidth=0.5,
            edgecolor="black");

# Grid line settings
ax5[0].grid(True, color="grey", alpha=0.1, linestyle="-")
ax5[1].grid(True, color="grey", alpha=0.1, linestyle="-")

# Tick parameters
ax5[0].tick_params(axis="y", left=False)
ax5[1].tick_params(axis="y", left=False)

# Plot border line settings
ax5[0].spines['top'].set_visible(False)
ax5[0].spines['right'].set_visible(False)
ax5[0].spines['bottom'].set_visible(True)
ax5[0].spines['left'].set_visible(True)
ax5[1].spines['top'].set_visible(False)
ax5[1].spines['right'].set_visible(False)
ax5[1].spines['bottom'].set_visible(True)
ax5[1].spines['left'].set_visible(False)

# Plot x-label
fig5.text(0.95, 0.001, "Salary Bracket ($USD)", fontproperties=prop, fontsize=13, ha='center', position=(0.52, -0.05))

# Plot title
fig5.text(0.95, 0.001, "Spanish and English Speaking Respondant Salary Count and Proportion", fontproperties=prop, fontsize=14, ha='center', position=(0.363, 0.95))

# Plot xticks
ax5[0].set_xticks(np.arange(19))
ax5[1].set_xticks(np.arange(19))
ax5[0].set_xticklabels(salary_labels, rotation=70, fontproperties=prop, fontsize=11)
ax5[1].set_xticklabels(salary_labels, rotation=70, fontproperties=prop, fontsize=11)

# Set y-label
ax5[0].set_ylabel("Number of 2018 Kaggle Survey Respondants", fontproperties=prop, fontsize=13, labelpad=20);