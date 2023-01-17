import pandas as pd
metadata_if_df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
print("Number of unique non-null journals: %d" % len(metadata_if_df.journal.value_counts()))



print("Number of PMIDs with non-null journals: %d" % len(metadata_if_df.loc[~metadata_if_df.journal.isna()]))



metadata_if_df.has_full_text.value_counts()
grouped_by_journal = metadata_if_df.groupby('journal')

full_text_articles = grouped_by_journal['has_full_text'].sum()
#Count total number of articles per journal



journal_total_counts = metadata_if_df.journal.value_counts()



journal_stats_df = pd.DataFrame(journal_total_counts)

journal_stats_df.rename(columns = {'journal' : 'total_articles'}, inplace=True)
#Count PMIDs per journal with full text and with no full text



full_text_counts_df = pd.DataFrame(full_text_articles)



article_count_full_text_df = pd.DataFrame.merge(full_text_counts_df, journal_stats_df, left_index=True, right_index=True)



article_count_full_text_df['no_full_text'] = article_count_full_text_df['total_articles'] - article_count_full_text_df['has_full_text']



pmids_no_full_text_df = metadata_if_df.loc[metadata_if_df.has_full_text == False]



pmids_no_full_text_list_df = pd.DataFrame(pmids_no_full_text_df.groupby(['journal'])['pubmed_id'].apply(list))
#Add list of PMIDs per journal with no full text



full_text_counts_df = pd.DataFrame.merge(pmids_no_full_text_list_df, article_count_full_text_df, left_index=True, right_index=True, how='right')



full_text_counts_df.rename(columns={'pubmed_id' : 'pmids_no_full_text'}, inplace=True)



full_text_counts_df['pmids_no_full_text'] = full_text_counts_df.pmids_no_full_text.fillna('None')



full_text_counts_df = full_text_counts_df[['total_articles', 'has_full_text', 'no_full_text', 'pmids_no_full_text']]
full_text_counts_df.to_csv('journal_no_full_text_counts_df_200324.csv')