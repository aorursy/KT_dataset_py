import pandas as pd
df = pd.read_csv('../input/github-repositories-analysis/Github_data.csv')
df.head()
df.shape
df[['topic', 'name', 'star', 'fork', 'watch', 'issue','topic_tag', 'discription_text', 'commits']]
# I have compiled only what I need. Save the data.
github_df = df[['topic', 'name', 'star', 'fork', 'watch', 'issue','topic_tag', 'discription_text', 'commits']]
# make a function that replace 'k' to '000'
def counts(x):
    rx = x.replace('k','000')
    if '.' in rx:
        rx = rx.replace('.','')
        rx = rx[:-1]
        return int(rx)
    return int(rx)
# test function counts()
github_df['star'].apply(counts)
github_df['fork'].apply(counts)
github_df['watch'].apply(counts)
# apply function counts() to data frame
github_df['star'] = github_df['star'].apply(counts)
github_df['fork'] = github_df['fork'].apply(counts)
github_df['watch'] = github_df['watch'].apply(counts)

github_df.head()
# Check the statistics summary to obtain the average value before analysis.
github_df.describe()
# check 'topic' column
github_df['topic'].drop_duplicates()
# It seems that 100 rows are extracted for each topic. Check it out.
github_df['topic'][90:110]
github_df['topic'][190:210]
# Check whether 100 rows were extracted randomly or only the upper repositories were extracted with statistics.
github_df[github_df['topic']=='Open-CV']
from IPython.display import Image
Image(filename='../input/opencv/Open-CV.png', height=280, width=800)
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about Data Science(10,000 or more stars -> 17)
github_df[github_df['topic']=='Data-Science'][:17]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about machine-Learning (10,000 or more stars -> 52)
github_df[github_df['topic']=='machine-Learning'][:52]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about Open-CV(10,000 or more stars -> 0)
github_df[github_df['topic']=='Open-CV'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about Computer-Vision(10,000 or more stars -> 8)
github_df[github_df['topic']=='Computer-Vision'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about GAN(10,000 or more stars -> 1)
github_df[github_df['topic']=='GAN'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about variational-encoder(10,000 or more stars -> 1)
github_df[github_df['topic']=='variational-encoder'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about Android-studio(10,000 or more stars -> 0)
github_df[github_df['topic']=='Android-studio'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about flutter(10,000 or more stars -> 4)
github_df[github_df['topic']=='flutter'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about java(10,000 or more stars -> 82)
github_df[github_df['topic']=='java'][:85]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about awesome(10,000 or more stars -> 0)
github_df[github_df['topic']=='awesome'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about javascript(10,000 or more stars -> 100+)
github_df[github_df['topic']=='javascript'][:100]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about c++(10,000 or more stars -> 100+)
github_df[github_df['topic']=='c++'][:100]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about Raspberry pi(10,000 or more stars -> 100+)
github_df[github_df['topic']=='Raspberry pi'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about Arduino(10,000 or more stars -> 4)
github_df[github_df['topic']=='Arduino'][:10]
'''
Topic list =    
0              Data-Science
100        machine-Learning
200                 Open-CV
300         Computer-Vision
400                     GAN
500     variational-encoder
600          Android-studio
700                 flutter
800                    java
900                 awesome
1000             javascript
1100                    c++
1200           Raspberry pi
1300                Arduino
1400                 sensor
'''
# Most Popular Repositories about sensor(10,000 or more stars -> 0)
github_df[github_df['topic']=='sensor'][:10]