import wordcloud

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

%pylab inline
df = pd.read_csv('../input/songdata.csv')

df.info()
allsongs = ' '.join(df.text).lower().replace('choru', '')

cloud = wordcloud.WordCloud(background_color='white',

                            max_font_size=100,

                            width=1000,

                            height=500,

                            max_words=300,

                            relative_scaling=.5).generate(allsongs)

plt.figure(figsize=(10,5))

plt.axis('off')

plt.savefig('allsongs.png')

plt.imshow(cloud);