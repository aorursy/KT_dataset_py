import pandas as pd

import matplotlib.pyplot as plt
file_path = '../input/facebook-antivaccination-dataset/posts_full.csv'

antivacc_data = pd.read_csv(file_path, index_col = 0)

antivacc_data.head()
list(antivacc_data.columns)
antivacc_data.timestamp.value_counts().nlargest(20).plot(figsize = (15, 5))

plt.legend()

plt.title('Timeline for Highest Frequency of Posts')

plt.show()
primary_sources = antivacc_data.article_host.value_counts().nlargest(20)

primary_sources.plot(figsize = (20, 5), kind = 'bar', rot = 30)

plt.title('Top 20 Third-Party Hosts of Anti-Vacc Content')

plt.legend()

plt.show()
data_length = antivacc_data.text_length

print('Mean Word Count:', data_length.mean())

print('Standard Deviation:', data_length.std())
data_length.nlargest(10).plot(kind = 'bar', figsize = (10, 5), rot = 0)

plt.title('Word Count of 10 Longest Articles')

plt.legend()

plt.show()
antivacc_data.anti_vax.value_counts().plot(kind = 'bar',figsize = (10, 5), rot = 0)

plt.title('Count of Nature of Content on Anti-Vax Campaign')

plt.legend()

plt.show()
page_names = antivacc_data.page_name.loc[antivacc_data.anti_vax == False]

page_names.dropna()

page_names.value_counts().nlargest(15).plot(kind = 'bar', figsize = (15, 5), rot = 40)

plt.legend()

plt.title('Top 15 Forums Against Anti-Vaccination Campaign')

plt.show()
page_names = antivacc_data.page_name.loc[antivacc_data.anti_vax == True]

page_names.dropna()

page_names.value_counts().nlargest(15).plot(kind = 'bar', figsize = (15, 5), rot = 60)

plt.legend()

plt.title('Top 15 Forums in Favour of Anti-Vaccination Campaigns')

plt.show()