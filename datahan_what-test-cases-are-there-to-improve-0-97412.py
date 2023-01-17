import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from collections import OrderedDict,defaultdict



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



pd.options.display.max_rows = 500

pd.options.display.max_columns = None

pd.set_option('display.max_colwidth', None)
group = pd.read_csv('../input/open-round1/Extra Material 2 - keyword list_with substring.csv')
titles = pd.read_csv('../input/open-round1/Keyword_spam_question.csv')
mismatch = pd.read_csv('../input/open-round1/Extra Material 3 - mismatch list.csv')
group.head()
titles.head()
titles.name = titles.name.str.lower()
titles.head()
mismatch.head()
mismatch_dic = {}

for col in mismatch:

    mismatch_dic[col] = [item for item in mismatch[col] if item == item] # numpy nans do not == themselves --> filtered



mismatch_dic
group.head()
group_split = group.copy()

group_split.Keywords = group_split.Keywords.str.split(',')
group_split
# keys not strip() yet! (don't be suprised to see leading space when using group_dict for debugging later)

group_dict = OrderedDict(zip(group_split.Group,group_split.Keywords))
len(group_dict)
rdict = {}



for index, keywords in reversed(group_dict.items()):

        for word in keywords:

            rdict[word.strip()] = index        # .strip!
rdict['notebook']
# drop spikes indicate keywords which got overwritten with a lower group

pd.Series(list(rdict.values()),index= list(rdict.keys())).plot()
pd.DataFrame({'word':list(rdict.keys()),'group':list(rdict.values())},index=range(len(rdict)))
substring_map = defaultdict(list)



for word in rdict:

     for word2 in rdict:

            if word != word2 and word in word2:

                substring_map[word].append(word2)

                print('{:<20}'.format(word),word2)
len(substring_map)

substring_map
substring_map_full = defaultdict(list)



for word in rdict:

     for word2 in rdict:

            if word != word2 and re.search(fr'\b{word}\b', word2):

                substring_map_full[word].append(word2)

                print('{:<20}'.format(word),word2)
len(substring_map_full)

substring_map_full
removed_keys = set(substring_map) - set(substring_map_full)

list(zip(removed_keys,[substring_map[key] for key in removed_keys]))
sum(titles[:10000].name.duplicated())

sum(titles.name.duplicated())
def search_func(title):

    result_groups = []

    for word in rdict:

        if word in title:

            if word in substring_map and any(superstring in title for superstring in substring_map[word]):   

                continue

            

            if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

                continue    

            result_groups.append(rdict[word])

    return result_groups



titles[:10].name.apply(search_func)
# result_df = titles[:1].copy()

# result_df['groups_found'] = ''



# for word in rdict:

#     contains_word = titles.name.str.contains(word)

#     print('contains_word_done')

    

#     # must check word in to prevent keyerrors

#     contains_superstring = word in substring_map and titles.name.str.contains('|'.join(substring_map[word]))

#     print('contains_superstring_done')

    

#     contains_mismatch = word in mismatch_dic and titles.name.str.contains('|'.join(mismatch_dic[word]))

#     print('contains_mismatch_done')

    

#     add_group = contains_word & ~contains_superstring & ~contains_mismatch

    

#     # to prepare for str.split to generate groups in list

#     new_string = ','+str(rdict[word])

#     result_df.loc[add_group,'groups_found'] = result_df.loc[add_group,'groups_found'] + new_string 
spaced_titles = titles.copy()

spaced_titles.name = ' '+spaced_titles.name+' '
# search_res_spaced = {}



# for index,title in enumerate(spaced_titles.name):

#     search_res_spaced[index] = []

#     for word in rdict:

#         if ' ' + word + ' ' in title:

#             #print(f'Title: {title}')

#             #print(f'Matched: {word}')

            

#             # prevent accessing non-existent key in substring_map (defaultdict) and causing empty list to generate

#             if word in substring_map and any(superstring in title for superstring in substring_map[word]):     # check eg 3 condition

#                 #print(f'{word}\n found in map: {substring_map[word]}\n')

#                 continue

            

#             if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

#                 #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                 continue    

                    

#             #print(f'{word}"s group added\n')

#             search_res_spaced[index].append(rdict[word])
# for index, groups_found in search_res_spaced.items():

#     search_res_spaced[index] = sorted(list(set(groups_found)))



# ans_df = pd.Series(list(search_res_spaced.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

# ans_df.to_csv('spaced_title_word.csv',index=False)
# sub_spaced = {}



# for index,title in enumerate(spaced_titles.name):

#     sub_spaced[index] = []

#     for word in rdict:

#         if ' ' + word + ' ' in title:

#             #print(f'Title: {title}')

#             #print(f'Matched: {word}')

            

#             # prevent accessing non-existent key in substring_map (defaultdict) and causing empty list to generate

#             if word in substring_map and any(' '+superstring+' ' in title for superstring in substring_map[word]):     # check eg 3 condition

#                 #print(f'{word}\n found in map: {substring_map[word]}\n')

#                 continue

            

#             if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

#                 #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                 continue    

                    

#             #print(f'{word}"s group added\n')

#             sub_spaced[index].append(rdict[word])
# for index, groups_found in sub_spaced.items():

#     sub_spaced[index] = sorted(list(set(groups_found)))



# spaced_tws_df = pd.Series(list(sub_spaced.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

# spaced_tws_df.to_csv('spaced_title_word_superstring.csv',index=False)
# sub_mismatch_spaced = {}



# for index,title in enumerate(spaced_titles.name):

#     sub_mismatch_spaced[index] = []

#     for word in rdict:

#         if ' ' + word + ' ' in title:

#             #print(f'Title: {title}')

#             #print(f'Matched: {word}')

            

#             # prevent accessing non-existent key in substring_map (defaultdict) and causing empty list to generate

#             if word in substring_map and any(' '+superstring+' ' in title for superstring in substring_map[word]):     # check eg 3 condition

#                 #print(f'{word}\n found in map: {substring_map[word]}\n')

#                 continue

            

#             if word in mismatch_dic and any(' '+mismatched_term+' ' in title for mismatched_term in mismatch_dic[word]):

#                 #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                 continue    

                    

#             #print(f'{word}"s group added\n')

#             sub_mismatch_spaced[index].append(rdict[word])
# for index, groups_found in sub_mismatch_spaced.items():

#     sub_mismatch_spaced[index] = sorted(list(set(groups_found)))



# spaced_twsm_df = pd.Series(list(sub_mismatch_spaced.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

# spaced_twsm_df.to_csv('spaced_title_word_substring_mismatch.csv',index=False)
# search_res = {}



# for index,title in enumerate(titles.name):

#     search_res[index] = []

#     for word in rdict:

#         # regex is 600x slower than `a in b`-> minimize regex use or wrap in many if conditions

#         #if re.search(f'\s{word}\s|^{word}\s|\s{word}$|^{word}$',title):  

        

#         if word in title:

#             if re.search(fr'\b{word}\b',title):

#                 #print(f'Title: {title}')

#                 #print(f'Matched: {word}')

            

#                 # prevent accessing non-existent key in substring_map (defaultdict) and causing empty list to generate

#                 if word in substring_map and any(superstring in title for superstring in substring_map[word]):     # check eg 3 condition

#                     #print(f'{word}\n found in map: {substring_map[word]}\n')

#                     continue



#                 if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

#                     #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                     continue    



#                 #print(f'{word}"s group added\n')

#                 search_res[index].append(rdict[word])
# for index, groups_found in search_res.items():

#     search_res[index] = sorted(list(set(groups_found)))



# regex_df = pd.Series(list(search_res.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

# regex_df.to_csv('regex_word_boundary.csv',index=False)
# search_res_full = {}



# for index,title in enumerate(titles.name):

#     search_res_full[index] = []

#     for word in rdict:

#         # regex is 600x slower than `a in b`-> minimize regex use or wrap in many if conditions

#         #if re.search(f'\s{word}\s|^{word}\s|\s{word}$|^{word}$',title):  

        

#         if word in title:

#             if re.search(fr'\b{word}\b',title):

#                 #print(f'Title: {title}')

#                 #print(f'Matched: {word}')

            

#                 # prevent accessing non-existent key in substring_map_full (defaultdict) and causing empty list to generate

#                 if word in substring_map_full and any(superstring in title for superstring in substring_map_full[word]):     # check eg 3 condition

#                     #print(f'{word}\n found in map: {substring_map_full[word]}\n')

#                     continue



#                 if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

#                     #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                     continue    



#                 #print(f'{word}"s group added\n')

#                 search_res_full[index].append(rdict[word])
# for index, groups_found in search_res_full.items():

#     search_res_full[index] = sorted(list(set(groups_found)))



# regex_df_full = pd.Series(list(search_res_full.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

# regex_df_full.to_csv('regex_word_boundary_fullmap.csv',index=False)
# full_super_ = {}



# for index,title in enumerate(titles.name):

#     full_super[index] = []

#     for word in rdict:

#         # regex is 600x slower than `a in b`-> minimize regex use or wrap in many if conditions

#         #if re.search(f'\s{word}\s|^{word}\s|\s{word}$|^{word}$',title):  

        

#         if word in title:

#             if re.search(fr'\b{word}\b',title):

#                 #print(f'Title: {title}')

#                 #print(f'Matched: {word}')

            

#                 # prevent accessing non-existent key in substring_map_full (defaultdict) and causing empty list to generate

#                 if word in substring_map_full and any(bool(re.search(fr'\b{superstring}\b',title)) for superstring in substring_map_full[word]):

#                 #if word in substring_map_full and any(superstring in title for superstring in substring_map_full[word]):     # check eg 3 condition

#                     #print(f'{word}\n found in map: {substring_map_full[word]}\n')

#                     continue



#                 if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

#                     #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                     continue    



#                 #print(f'{word}"s group added\n')

#                 full_super[index].append(rdict[word])
# for index, groups_found in full_super.items():

#     full_super[index] = sorted(list(set(groups_found)))



# regex_df_full_super = pd.Series(list(full_super.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

# regex_df_full_super.to_csv('regex_word_boundary_fullmap_super.csv',index=False)
full_super_mismatch = {}



for index,title in enumerate(titles.name):

    full_super_mismatch[index] = []

    for word in rdict:

        # regex is 600x slower than `a in b`-> minimize regex use or wrap in many if conditions

        #if re.search(f'\s{word}\s|^{word}\s|\s{word}$|^{word}$',title):  

        

        if word in title:

            if re.search(fr'\b{word}\b',title):

                #print(f'Title: {title}')

                #print(f'Matched: {word}')

            

                # prevent accessing non-existent key in substring_map_full (defaultdict) and causing empty list to generate

                if word in substring_map_full and any(bool(re.search(fr'\b{superstring}\b',title)) for superstring in substring_map_full[word]):

                #if word in substring_map_full and any(superstring in title for superstring in substring_map_full[word]):     # check eg 3 condition

                    #print(f'{word}\n found in map: {substring_map_full[word]}\n')

                    continue

                

                if word in mismatch_dic and any(bool(re.search(fr'\b{mismatched_term}\b',title)) for mismatched_term in mismatch_dic[word]):

                #if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

                    #print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

                    continue    



                #print(f'{word}"s group added\n')

                full_super_mismatch[index].append(rdict[word])
for index, groups_found in full_super_mismatch.items():

    full_super_mismatch[index] = sorted(list(set(groups_found)))



regex_df_full_super_mismatch = pd.Series(list(full_super_mismatch.values())).to_frame().reset_index().rename(columns={0:'groups_found'})

regex_df_full_super_mismatch.to_csv('regex_word_boundary_fullmap_super_mismatch.csv',index=False)
plt.hist(list(map(len,full_super.values())))  # list() because RuntimeError: matplotlib does not support generators as input
# regex_df_full_super_mismatch =  pd.read_csv('regex_word_boundary_fullmap_super_mismatch.csv')

# regex_df_full_super  = pd.read_csv('regex_word_boundary_fullmap_super.csv')
# sum(regex_df_full_super.groups_found!=regex_df_full_super_mismatch.groups_found)
# regex_df_full_super[regex_df_full_super.groups_found!=regex_df_full_super_mismatch.groups_found].sample(5,random_state=2)

# regex_df_full_super_mismatch[regex_df_full_super.groups_found!=regex_df_full_super_mismatch.groups_found].sample(5,random_state=2)
# titles.loc[	316836,'name']



# # select a word from this list and assign to keyword below to check what superstrings/mismatch are there

# print('keywords in group:', group_dict[239])  



# keyword = 'jam'



# print('superstrings: ',substring_map_full[keyword])

# print('mismatched words: ',mismatch_dic[keyword])
# index = 654229



# for index,title in enumerate(titles[index:index+1].name):

#     #search_res[index] = []

#     for word in rdict:

#         # regex is 600x slower than `a in b`-> minimize regex use or wrap in many if conditions

#         #if re.search(f'\s{word}\s|^{word}\s|\s{word}$|^{word}$',title):  

        

#         if word in title:

#             if re.search(fr'\b{word}\b',title):

#                 print(f'Title: {title}')

#                 print(f'Matched: {word}')

#                 print(f'rdict group number: {rdict[word]}')

                

            

#                 # prevent accessing non-existent key in substring_map (defaultdict) and causing empty list to generate

#                 if word in substring_map and any(superstring in title for superstring in substring_map[word]):     # check eg 3 condition

#                     print(f'{word} found in map: {substring_map[word]}\n')

#                     continue



#                 if word in mismatch_dic and any(mismatched_term in title for mismatched_term in mismatch_dic[word]):

#                     print(f'{word} found in mismatch_dic: {mismatch_dic[word]} \n')

#                     continue    



#                 #print(f'{word}"s group added\n')

#                 #search_res[index].append(rdict[word])
#df1 = regex_df_full

#df2 = regex_df_full_super



# removal_prevention = df1[df1.groups_found!=df2.groups_found]

# no_removal_prevention = df2[df1.groups_found!=df2.groups_found]



# all(removal_prevention.groups_found.str.len()> no_removal_prevention.groups_found.str.len())