import pandas as pd



(pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv', 

             low_memory=False, skiprows=1) # read questions with multiple choices

   .filter(regex='^((?! - Text).)*$') # remove useless free text columns

   .filter(regex='^(?!Duration)') # remove duration column

   .melt(var_name='question', value_name='answer') # transform data from wide to long data format

   .dropna(subset=['answer']) # remove any lines that don't contain any answer

   .assign(question_type = lambda x: x['question'].str.split('-', 1) 

                                                  .str[0]

                                                  .str.split('(:|\?)')

                                                  .str[0]

                                                  .str.strip()) # get shared question part (for grouping further)

   .groupby('question_type')['answer'] # group by shared question parts (question type)

   .value_counts() # calculate counts of every value for specific question type

   .rename('count') # rename series

   .reset_index() # reset index for getting access to index fields for following steps

   .groupby('question_type', as_index=False) # group by question type for plotting

   .apply(lambda data: data.sort_values('count', ascending=True)

                           .tail(20)

                           .plot.barh(y='count', x='answer', 

                                      title=data['question_type'][0], 

                                      figsize=(10, 0.7 * len(data.tail(20))))) # plot data for each question type

);