import numpy as np # linear algebra

import os

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def doi_url(d):

    if '"' in d:

        d = str.split(d,'"')[1]

    if d.startswith('http://'):

        return d

    if d.startswith('https://'):

        return d

    elif d.startswith('doi.org'):

        return f'http://{d}'

    else:

        return f'http://doi.org/{d}'
# read CORD-19 metadata, set up subset dfs for matching 

metadata_df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

# clean url - should be a doi

metadata_df.url = metadata_df.url.fillna('').apply(doi_url)



metadata_doi_df = metadata_df[metadata_df.columns.intersection(['cord_uid', 'doi'])]

metadata_doi_df = metadata_doi_df.rename(columns = {'cord_uid':'doi_cord_uid', 'doi':'doi_metadata'})

metadata_doi_df['doi_metadata'] = "https://doi.org/" + str(metadata_doi_df['doi_metadata'])

# print(metadata_doi_df)

metadata_title_df = metadata_df[metadata_df.columns.intersection(['cord_uid', 'title'])]

metadata_title_df = metadata_title_df.rename(columns = {'cord_uid':'title_cord_uid', 'title':'title_metadata'})

metadata_url_df = metadata_df[metadata_df.columns.intersection(['cord_uid', 'url'])]

# explode delimited url values onto separate rows

metadata_url_df.assign(url=metadata_url_df['url'].astype(str).str.split(';')).explode('url')

metadata_url_df = metadata_url_df.rename(columns = {'cord_uid':'url_cord_uid', 'url':'url_metadata'})



metadata_df = None
# Loop over input data files, deriving metadata



all_papers_df = None



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

#         print(os.path.join(dirname, filename))

        if (( "/target_tables/" in dirname or "/cord-19-task-csv-exports/" in dirname 

            or "/coronawhy-task-ties-patient-descriptions" in dirname

            or "/covid-19-temperature-and-humidity-summary-tables" in dirname)

            and filename.endswith(".csv")):

            this_file_df = pd.read_csv(os.path.join(dirname, filename))

            

            # conform filename

            if filename == "temperature_or_humidity.csv":

                filename = "How does temperature and humidity affect the transmission of 2019-nCoV_.csv"



        # add metadata columns

            ML_Notebook = str(str.split(dirname,"/input/")[1])

            ML_Notebook = str(str.split(ML_Notebook,"/")[0])

            this_file_df["ML Notebook"] = ML_Notebook



            if "CORD-19-research-challenge" in ML_Notebook:

                ML_Author = "Kaggle Community"

                ML_Notebook_URL = "https://kaggle.com/allen-institute-for-ai/" + ML_Notebook

                if "/0_" in dirname or "/unsorted" in dirname:

                    break

                else:

                    None

            elif "cord-19-task-csv-exports" in ML_Notebook:

                ML_Author = "David Mezzetti"

                ML_Notebook_URL = "https://kaggle.com/davidmezzetti/" + ML_Notebook

            elif "coronawhy-task-ties-patient-descriptions" in ML_Notebook: 

                ML_Author = "CoronaWhy Team Task-TIES"

                ML_Notebook_URL = "https://kaggle.com/crispyc/" + ML_Notebook

            elif "covid-19-temperature-and-humidity-summary-tables" in ML_Notebook: 

                ML_Author = "Javier Sastre et.al"

                ML_Notebook_URL = "https://kaggle.com/javiersastre/" + ML_Notebook

            else:

                ML_Author = ML_Notebook

                ML_Notebook_URL = ""



            this_file_df["ML Author"] = ML_Author

            this_file_df["ML Notebook URL"] = ML_Notebook_URL

                

            this_file_df["File Name"] = filename

            print(ML_Author + " - " + filename)



    # conform column types and names

            this_file_df.rename(columns={'Study Type ': 'Study Type'}, inplace=True)

            this_file_df.rename(columns={'Study type': 'Study Type'}, inplace=True)

            this_file_df.rename(columns={'Study':'Title'}, inplace=True)

            this_file_df.rename(columns={'Link':'URL'}, inplace=True)

            this_file_df.rename(columns={'Study Link':'URL'}, inplace=True)

            this_file_df.rename(columns={'Study link':'URL'}, inplace=True)



            # clean url - should be a doi

            this_file_df.URL = this_file_df.URL.fillna('').apply(doi_url)



            this_file_df.rename(columns={'Discharge vs. death?': 'Discharged vs. death?'}, inplace=True)

            allcolumns = list(this_file_df)

            this_file_df[allcolumns] = this_file_df[allcolumns].fillna('')

            this_file_df[allcolumns] = this_file_df[allcolumns].astype(str)



            # engineer new attributes

            this_file_df_copy = this_file_df.copy()

            for idx in this_file_df_copy.index:

                

                # clean URL

                try:

                    if not "http" in this_file_df_copy.at[idx,'URL']:

                        this_file_df.at[idx,'URL'] = "https://" + this_file_df_copy.at[idx,'URL']

                except:

                    None

                

                # append Journal to Title

                Title = ""

                try:

                    Title = this_file_df_copy.at[idx,'Title']

                    Title = Title + " (" + this_file_df_copy.at[idx,'Journal'] + ")"

                except:

                    None

                this_file_df.at[idx,'Title'] = Title

                

                # clean Sample Size

                Sample_Size = ""

                try:

                    Sample_Size = str.replace(str(this_file_df_copy.at[idx,'Sample Size']), "n=", "")

                    this_file_df.at[idx,'Sample Size'] = Sample_Size

                except:

                    this_file_df.at[idx,'Sample Size'] = Sample_Size



                # engineer Sample - patients

                Sample_subjects = ""

                try:

                    Sample_subjects = str.strip(str(this_file_df_copy.at[idx,'Sample Size']))

    #                 print("in: " + Sample_subjects)

                    Sample_subjects = str.replace(Sample_subjects, " subjects", " patients")

                    Sample_subjects = str.split(Sample_subjects, " subjects")[0]

                    Sample_subjects = str(str.split(Sample_subjects," ")[-2])

    #                 print("out: " + Sample_subjects)

                    this_file_df.at[idx,'Sample Subjects'] = Sample_subjects

                except:

    #                 print("except: " + Sample_subjects)

                    this_file_df.at[idx,'Sample Subjects'] = Sample_subjects



                Severe_Raw =""

                try:

                    Severe_Raw = str(this_file_df_copy.at[idx,'Severe'])

                    Severe_Raw = str.replace(Severe_Raw, '\s+', ' ',regex=True )

                except:

                    None



                # engineer Severe Metric, e.g. OR, RR

                Severe_Metric = ""

                try:

                    Severe_Metric = str.strip(Severe_Raw)

                    Severe_Metric = str.strip(str.replace(str.split(Severe_Metric," ")[0],":",""))

                    Severe_Metric = str.strip(str.split(Severe_Metric,"=")[0])

                    this_file_df.at[idx,'Severe Metric'] = Severe_Metric

                except:

                    None



                # engineer Severe Value, e.g. 1.07 or whatever the ratio is.

                Severe_Value = ""

                try:

                    Severe_Value = str.strip(str.replace(str.replace(str.replace(Severe_Raw, ":", " "), "=", " "),"  ", " "))

                    Severe_Value = str.split(Severe_Value," ")[1]

                    this_file_df.at[idx,'Severe Value'] = Severe_Value

                except:

                    None



                # engineer Severe Method

                Severe_Method = ""

                try:

                    Severe_Method = str.strip(Severe_Raw)

                    Severe_Method = str.strip(str.replace(str.replace(Severe_Method, Severe_Value ,""), ":", ""))

                    Severe_Method = str.split(Severe_Method, " ")[0]           

                    this_file_df.at[idx,'Severe Method'] = Severe_Method

                except:

                    None



                # engineer Severe Label

                Critical_Only_Footnote = ""

                try:

                    Critical_Only = str.lower(this_file_df_copy.at[idx,'Critical only'])

                    if Critical_Only == "y":

                        Critical_Only_Footnote = "‡ "

                except:

                    None

                    

                Discharged_vs_Death = ""

                Discharged_vs_Death_Footnote = ""

                try:

                    Discharged_vs_Death = str.lower(this_file_df_copy.at[idx,'Discharged vs. death?'])

                except:

                    None

                if Discharged_vs_Death == "y":

                    Discharged_vs_Death_Footnote = "⸸ "

                

                Severe_Label = ""

                this_file_df.at[idx,'Severe Label'] = Severe_Label

                

                try:

                    if Severe_Raw != "":

                        Severe_Label = Severe_Raw

                        Severe_Label = Critical_Only_Footnote + Discharged_vs_Death_Footnote + Severe_Label

                        if "not adjusted" in str.lower(this_file_df_copy.at[idx,'Severe Adjusted']):

                            Severe_Label = "§ " + Severe_Label

                        if "calculated" in str.lower(this_file_df_copy.at[idx,'Severe Calculated']):

                            Severe_Label = "† " + Severe_Label

                        if this_file_df_copy.at[idx,'Severe lower bound'] != None:

                            Severe_Label = Severe_Label + " (95% CI: " + str(this_file_df_copy.at[idx,'Severe lower bound'])

#                         print("debug 1: " + Severe_Label) if this_file_df_copy.at[idx,'URL'] == "https://doi.org/10.1101/2020.04.24.20078006" else None

                        if this_file_df_copy.at[idx,'Severe upper bound'] != None:

                            Severe_Label = Severe_Label + "-" + str(this_file_df_copy.at[idx,'Severe upper bound']) + ")"

                        if str(this_file_df_copy.at[idx,'Severe p-value']) > "":

                            if str.isdigit(str(this_file_df_copy.at[idx,'Severe p-value'])[:1]):

                                Severe_Label = Severe_Label + " p=" + str(this_file_df_copy.at[idx,'Severe p-value'])

                            else:

                                Severe_Label = Severe_Label + " p" + this_file_df_copy.at[idx,'Severe p-value']

                            Severe_Label = Severe_Label + ")" 



                        Severe_Label = ( Severe_Label + ", " + this_file_df_copy.at[idx,'Severe Significant'] + ", " + 

                            this_file_df_copy.at[idx,'Severe Adjusted'] + ", " + 

                            this_file_df_copy.at[idx,'Severe Calculated'] )

#                         print("debug 2: " + Severe_Label) if this_file_df_copy.at[idx,'URL'] == "https://doi.org/10.1101/2020.04.24.20078006" else None

                        this_file_df.at[idx,'Severe Label'] = Severe_Label

        #                 print (Severe_Label)

                    else:

                        None

                except:

                    this_file_df.at[idx,'Severe Label'] = Severe_Label

                

                Severe_Label_Background_Color = ""

                try:

                    if this_file_df_copy.at[idx,'Severe Significant'] == "":

                        Severe_Label_Background_Color = ""

                    elif "not significant" in str.lower(this_file_df_copy.at[idx,'Severe Significant']):

                        Severe_Label_Background_Color = "#F4C7C3"

                    else:

                        Severe_Label_Background_Color = "#B7E1CD"

                except:

                    None

                this_file_df.at[idx,'Severe Label Background Color'] = Severe_Label_Background_Color

#                 print("debug 1: " + Severe_Label_Background_Color) if this_file_df_copy.at[idx,'URL'] == "https://doi.org/10.1101/2020.04.24.20078006" else None

                

                Fatality_Raw =""

                try:

                    Fatality_Raw = str(this_file_df_copy.at[idx,'Fatality'])

                    Fatality_Raw = str.replace(Fatality_Raw, '\s+', ' ',regex=True )

                except:

                    None



                # engineer Fatality Metric

                Fatality_Metric = ""

                try:

                    Fatality_Metric = str.strip(Fatality_Raw)

                    Fatality_Metric = str.strip(str.replace(str.split(Fatality_Metric," ")[0],":",""))

                    Fatality_Metric = str.strip(str.split(Fatality_Metric, "=")[0])

                    this_file_df.at[idx,'Fatality Metric'] = Fatality_Metric

                except:

                    None



                # engineer Fatality Value

                Fatality_Value = Fatality_Raw

                try:

                    Fatality_Value = str.strip(str.replace(str.replace(str.replace(Fatality_Value, ":", " "), "=", " "),"  ", " "))

                    Fatality_Value = str.split(str.strip(Fatality_Value)," ")[1]

                    this_file_df.at[idx,'Fatality Value'] = Fatality_Value

                except:

                    None



                # engineer Fatality Method

                Fatality_Method = ""

                try:

                    Fatality_Method = str.strip(Fatality_Raw)

                    Fatality_Method = str.strip(str.replace(str.replace(Fatality_Method, Fatality_Value ,""), ":", ""))

                    Fatality_Method = str.split(Fatality_Method, " ")[0]           

                    this_file_df.at[idx,'Fatality Method'] = Fatality_Method

                except:

                    this_file_df.at[idx,'Fatality Method'] = Fatality_Method



                # engineer Fatality Label

                Fatality_Label = ""

                this_file_df.at[idx,'Fatality Label'] = Fatality_Label

                try:

                    if Fatality_Raw != "":

                        Fatality_Label = Fatality_Raw 

                        Fatality_Label = Critical_Only_Footnote + Discharged_vs_Death_Footnote + Fatality_Label

                        if "not adjusted" in str.lower(this_file_df_copy.at[idx,'Fatality Adjusted']):

                            Fatality_Label = "§ " + Fatality_Label

                        if "calculated" in str.lower(this_file_df_copy.at[idx,'Fatality Calculated']):

                            Fatality_Label = "† " + Fatality_Label

                        if this_file_df_copy.at[idx,'Fatality lower bound'] != None:

                            Fatality_Label = Fatality_Label + " (95% CI: " + str(this_file_df_copy.at[idx,'Fatality lower bound'])

                        if this_file_df_copy.at[idx,'Fatality upper bound'] != None:

                            Fatality_Label = Fatality_Label + "-" + str(this_file_df_copy.at[idx,'Fatality upper bound']) + ")"

                        if str(this_file_df_copy.at[idx,'Fatality p-value']) > "":

                            if str.isdigit(str(this_file_df_copy.at[idx,'Fatality p-value'])[:1]):

                                Fatality_Label = Fatality_Label + " p=" + str(this_file_df_copy.at[idx,'Fatality p-value'])

                            else:

                                Fatality_Label = Fatality_Label + " p" + this_file_df_copy.at[idx,'Fatality p-value']

                            Fatality_Label = Fatality_Label + ")" 



                        if str(this_file_df_copy.at[idx,'Fatality Significant']) > "":

                            Fatality_Label = Fatality_Label + ", " + this_file_df_copy.at[idx,'Fatality Significant']  

                        if str(this_file_df_copy.at[idx,'Fatality Adjusted']) > "":

                            Fatality_Label = Fatality_Label + ", " + this_file_df_copy.at[idx,'Fatality Adjusted']  

                        if str(this_file_df_copy.at[idx,'Fatality Calculated']) > "":

                            Fatality_Label = Fatality_Label + ", " + this_file_df_copy.at[idx,'Fatality Calculated']  



        #                 print (Fatality_Label)

                        this_file_df.at[idx,'Fatality Label'] = Fatality_Label

                    else:

                        None

                except:

                    this_file_df.at[idx,'Fatality Label'] = Fatality_Label



                Fatality_Label_Background_Color = ""

                try:

                    if this_file_df_copy.at[idx,'Fatality Significant'] == "":

                        Fatality_Label_Background_Color = ""

                    elif "not significant" in str.lower(this_file_df_copy.at[idx,'Fatality Significant']):

                        Fatality_Label_Background_Color = "#F4C7C3"

                    else:

                        Fatality_Label_Background_Color = "#B7E1CD"

                except:

                    None

                this_file_df.at[idx,'Fatality Label Background Color'] = Fatality_Label_Background_Color





    #       append this file to the collection df

            all_papers_df = pd.concat([this_file_df, all_papers_df], ignore_index=True, sort=False)
all_papers_merge_df = all_papers_df[all_papers_df.columns.intersection(['Title','URL'])]



# doi Merge - Cartesian product by introducing key(self join)

combined_df = all_papers_merge_df.merge(metadata_doi_df, how="left", left_on=['URL'], right_on=['doi_metadata'])

combined_df = combined_df.merge(metadata_url_df, how="left", left_on=['URL'], right_on=['url_metadata'])

combined_df = combined_df.merge(metadata_title_df, how="left", left_on=['Title'], right_on=['title_metadata'])

combined_df = combined_df[combined_df.columns.intersection(['doi_cord_uid', 'doi_metadata', 'url_cord_uid', 'url_metadata', 'title_cord_uid', 'title_metadata'])]

# print(combined_df)



all_papers_df = all_papers_df.merge(combined_df, left_index=True, right_index=True)

# print(all_papers_df)



# write out the all_papers df

all_papers_df['all papers index'] = all_papers_df.index

all_papers_df.to_csv('all_papers.csv', index = False)

print("Wrote the all_papers.csv file with: " + str(len(all_papers_df.index)) + " rows of table data.")            



# create value-pair version of file. Preserve common columns and pivot every other column into Attribute and Value

all_papers_df['Study Type - Copy'] = all_papers_df['Study Type']

all_papers_value_pairs_df = (all_papers_df.set_index(['all papers index', "ML Author", "ML Notebook", "ML Notebook URL", "File Name", "Date", "Title", "URL", "Journal", "Study Type - Copy"])

                             .stack().reset_index())

all_papers_value_pairs_df.rename(columns={'level_10': 'Attribute'}, inplace=True)

all_papers_value_pairs_df = all_papers_value_pairs_df.query('Attribute != "Unnamed: 0"')

all_papers_value_pairs_df.rename(columns={'Study Type - Copy':'Study Type'}, inplace=True)

# print(all_papers_value_pairs_df)

all_papers_value_pairs_df.to_csv('all_papers_value_pairs.csv', index = False)



print("Wrote the all_papers_value_pairs.csv file with: " + str(len(all_papers_value_pairs_df.index)) + " rows of value-pair data.")            

print("Finished!")