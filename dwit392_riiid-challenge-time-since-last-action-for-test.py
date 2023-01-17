def user_state(test_df, last_record):

    #The first part of the function calculates the time since last action and updates last_record with the new timestamp

    action_time = np.empty(len(test_df))

    for i in range(len(test_df)):

        if test_df.user_id.iloc[i] in last_record.user_id.values: 

            #check if the user_id is in the DF ... else add it

            new_time = test_df.timestamp.iloc[i]

            #new timestamp

            old_time = last_record.loc[last_record.user_id == test_df.user_id.iloc[i], 'timestamp'].values[0]

            #looking up old timestamp

            if (new_time > old_time):

                #is the new timestamp greater

                action_time[i] = new_time - old_time

                #then calculate time differential

                last_record.loc[last_record.user_id == test_df.user_id.iloc[i], 'timestamp'] = new_time

                #update timestamp value

            

            elif (new_time == old_time):

                #else is it equal

                try: #try just in case there is some index nonsense case

                    action_time[i] = action_time[i-1] #fill w/ prior value

                    

                except: 

                    action_time[i] = time_since_median #this shouldn't matter

            

            else:

                #this shouldn't happen

                action_time[i] = time_since_median

                

        else:

            #add new row to DF and fill value with 

            action_time[i] = time_since_median

            last_record.loc[len(last_record)] = test_df[['user_id','timestamp']].iloc[i]

            

    return action_time, last_record