import pandas as pd
import numpy as np
import altair as alt
from sklearn.feature_extraction.text import CountVectorizer
import ast #convert columns wiht lists to lst
import os
#set enviorment

#open metadata into pandas df
df = pd.read_csv('../input/mesminer/df_mesminer_final.csv', error_bad_lines=False, encoding='ISO-8859-1',index_col=0)

df['Biomedical_Entities'] =  df['Biomedical_Entities'].apply(ast.literal_eval)

df['genes_'] =  df['Genes'].apply(ast.literal_eval)
df['drugs_'] =  df['Drugs'].apply(ast.literal_eval)

df.head(2)#have a look at the dataframe
#This cell allows you to perform a search whithin the database
df_search = df
#you can create a search of papers that contain key biomedical entities, 
#add them in the list below as ['liver','neuron','X']

#remove '#' in selection and df = ... to apply the search

#you can create a search of papers that contain keywords, add them in the list
#selection = ['liver']
#df_search = df_search[pd.DataFrame(df_search.Biomedical_Entities.tolist()).isin(selection).any(1)]

#you can create a search of genes of interest, add them in the list
#selection = ['vim']
#df_search = df_search[pd.DataFrame(df_search.genes_.tolist()).isin(selection).any(1)]


#you can create a search of drugs of interest, add them in the list

#selection = ['cysteine']
#df_search = df_search[pd.DataFrame(df_search.drugs_.tolist()).isin(selection).any(1)]


#you can create only look at preprints or peer reviewed papers
#selection = ['Peer-Review']
#df = df[pd.DataFrame(df.preprint.tolist()).isin(selection).any(1)]


#df_search
#get top n words
def get_top_n_genes(corpus, n=20):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]#Convert most freq words to dataframe for plotting bar plot

#Check if the subselection contains any genes
def check_if_empty(list_of_lists):
    lst =[]

    for elem in list_of_lists:
       
        if len(elem) == 0:
            lst.append(False)
        else:
            lst.append(True)
    if True in lst:
        return True
    else:
        return False
    

#visualises a column of df the using altair of the top items in a column
def viz_data(df,col,n=20):
    if n == 0:
        n = 20
        print('You did asked for 0 genes so I am showing you the top 20!')

    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
            flat_list = [item for sublist in lst for item in sublist]
            top_words = get_top_n_genes(lst, n)
            top_df = pd.DataFrame(top_words)
            top_df.columns=[col, "Freq_gene"]
            plot = alt.Chart(top_df).mark_bar().encode(
    x = alt.X(col,sort ='-y', axis=alt.Axis( title = col)),
    y = alt.Y('Freq_gene', axis=alt.Axis( title = 'Frequency Count'))
    ).properties(title = "Frequency of the top {} {}".format(str(n),col), width = 200
               )

            return plot
    else:
        return False
    
#visualise top genes and drugs 
def viz_multiple_columns(df,lst,n=20):
    plots = []
    for el in lst:
        plt = viz_data(df,el,n)
        if plt != False:
            plots.append(plt)
            #plots.append('|')
    
    a = plots[0]
    for i in range(1,len(plots)):
         a = a | plots[i]
        
    return a
#n = int(input("Please enter a number: "))

viz_multiple_columns(df,['Genes','Drugs'],20)
#This function returns a df with the frequency of the top genes or drugs
#for downstream applications

def get_df(df,col,n=20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
            flat_list = [item for sublist in lst for item in sublist]
            top_words = get_top_n_genes(lst, n)
            top_df = pd.DataFrame(top_words)
            top_df.columns=[col, "Freq_gene"]

            return top_df
    else:
        return False
#Use it for Genes or Drugs
top_words_df = get_df(df,'Genes', n=20)
top_words_df.head()
#you can save your genes or drugs as a csv file if you want!
#top_words_df.to_csv('top_words_df.csv',index=False)

#visualises a column of the df using altair of the top items in a column

def viz_data_up_reg(df,col,n=20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
            top_words = get_top_n_genes(lst, n)
            top_df = pd.DataFrame(top_words)
            top_df.columns=[col, "Freq_gene"]
            plot = alt.Chart(top_df).mark_bar().encode(
    x = alt.X(col,sort ='-y', axis=alt.Axis( title = col)),
    y = alt.Y('Freq_gene', axis=alt.Axis( title = 'Frequency Count'))
    ).properties(title = "Frequency of the top {} {}".format(str(n),col), width = 200
               )

            return plot
    else:
        return False
    #first we need to check if the lists are empty

#visualise regulation

#Regulation is calculated using nlp techniques. Currently it is a simplistic based on dentification of common terms used to describe gene regulation in the surrounding text of an identified gene.
#More details in the Github page
def viz_multiple_columns_reg(df,lst,n=20):
    
    if n == 0:
        n = 20
        print('You did asked for 0 genes so I am showing you the top 20!')

    plots = []
    for el in lst:
        plt = viz_data_up_reg(df,el,n)
        if plt != False:
            plots.append(plt)
            #plots.append('|')
    
    a = plots[0]
    for i in range(1,len(plots)):
         a = a | plots[i]
        
    return a
#n = int(input("Please enter a number: "))


viz_multiple_columns_reg(df,['Genes_Upregulated','Genes_Downregulated','Genes_Nonregulated'],20)
#This function returns a df with the frequency of the regulation
#for downstream applications

def get_df(df,col,n=20):
    sr = df[col].dropna()
    lst = sr.tolist()
    check = check_if_empty(lst)
    if check == True:
            top_words = get_top_n_genes(lst, n)
            top_df = pd.DataFrame(top_words)
            top_df.columns=[col, "Freq_gene"]

            return top_df
    else:
        return False
#Use it for reguraion
top_words_df = get_df(df,'Genes_Upregulated', n=20)
top_words_df.head()
#you can save your genes or drugs as a csv file if you want!
#top_words_df.to_csv('top_words_df.csv',index=False)