import pandas as pd

from collections import Counter

import seaborn as sns



def count_hashtags(inp_fpath):

    def flatten_list(inp):

        return([item for sublist in inp for item in sublist])

    file_contents = pd.read_csv(inp_fpath)

    hash_tag_list = file_contents.hashtags.dropna().tolist()

    # ignore casing differences

    hash_tag_list = [s.lower() for s in hash_tag_list]

    # remove quotes in original list of hashtags

    hash_tag_list = [s.replace('"', "") for s in hash_tag_list]

    hash_tag_list = flatten_list([s.split(" ") for s in hash_tag_list])

    word_counts = Counter(hash_tag_list)

    word_counts_df = pd.DataFrame(list(word_counts.items()), 

                                  columns=['Hash Tag', 'Count']).set_index(["Hash Tag"])

    return(word_counts_df)



aug15 = count_hashtags('../input/aug15_sample.csv')

aug16 = count_hashtags('../input/aug16_sample.csv')

aug17 = count_hashtags('../input/aug17_sample.csv')



aug15['Date'] = 'Aug 15, 2017'

aug16['Date'] = 'Aug 16, 2017'

aug17['Date'] = 'Aug 17, 2017'



hashtag_counts = pd.concat([aug15, aug16, aug17])



# Filter to commonly used hash tags for plotting

total_uses = hashtag_counts.groupby(['Hash Tag']).sum()['Count']

popular_hashtags = hashtag_counts[total_uses > 350]

to_plot = popular_hashtags.reset_index()



g = sns.factorplot(x="Hash Tag", y="Count", hue="Date", data=to_plot, size=6, kind="bar", aspect=2)


