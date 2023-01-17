def cnt_miss(df):

    cnt = 0

    for e in range(len(df)):

        x = df['event_data'].iloc[e]

        y = json.loads(x)['misses']

        cnt += y

    return cnt



def get_4020_acc(df,counter_dict):

    

    for e in ['Cauldron Filler (Assessment)','Bird Measurer (Assessment)','Mushroom Sorter (Assessment)','Chest Sorter (Assessment)']:

        

        Assess_4020 = df[(df.event_code == 4020) & (df.title==activities_map[e])]   

        true_attempts_ = Assess_4020['event_data'].str.contains('true').sum()

        false_attempts_ = Assess_4020['event_data'].str.contains('false').sum()



        measure_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0

        counter_dict[e+"_4020_accuracy"] += (counter_dict[e+"_4020_accuracy"] + measure_assess_accuracy_) / 2.0

    

    return counter_dict
def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    

    train['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), train['type'], train['world']))

    test['type_world'] = list(map(lambda x, y: str(x) + '_' + str(y), test['type'], test['world']))

    all_type_world = list(set(train["type_world"].unique()).union(test["type_world"].unique()))

    

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map,all_type_world
# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    assess_4020_acc_dict = {'Cauldron Filler (Assessment)_4020_accuracy':0,

                                'Mushroom Sorter (Assessment)_4020_accuracy':0,

                                'Bird Measurer (Assessment)_4020_accuracy':0,

                                'Chest Sorter (Assessment)_4020_accuracy':0 }

    

    game_time_dict = {'Clip_gametime':0, 'Game_gametime':0, 'Activity_gametime':0, 'Assessment_gametime':0}

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    

    # Newly added features

    accumulated_game_miss = 0

    Cauldron_Filler_4025 = 0

    mean_game_round = 0

    mean_game_duration = 0 

    mean_game_level = 0

    Assessment_mean_event_count = 0

    Game_mean_event_count = 0

    Activity_mean_event_count = 0

    chest_assessment_uncorrect_sum = 0

    

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

    type_world_count: Dict[str, int] = {w_eve: 0 for w_eve in all_type_world}

        

    # last features

    sessions_count = 0

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

                    

        if session_type=="Activity":

            Activity_mean_event_count = (Activity_mean_event_count + session['event_count'].iloc[-1])/2.0

            

        if session_type=="Game":

            

            Game_mean_event_count = (Game_mean_event_count + session['event_count'].iloc[-1])/2.0

            

            game_s = session[session.event_code == 2030]   

            misses_cnt = cnt_miss(game_s)

            accumulated_game_miss += misses_cnt

            

            try:

                game_round = json.loads(session['event_data'].iloc[-1])["round"]

                mean_game_round =  (mean_game_round + game_round)/2.0

            except:

                pass



            try:

                game_duration = json.loads(session['event_data'].iloc[-1])["duration"]

                mean_game_duration = (mean_game_duration + game_duration) /2.0

            except:

                pass

            

            try:

                game_level = json.loads(session['event_data'].iloc[-1])["level"]

                mean_game_level = (mean_game_level + game_level) /2.0

            except:

                pass

            

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(game_time_dict.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            features.update(assess_4020_acc_dict.copy())

            features.update(type_world_count.copy())

            

            features['installation_session_count'] = sessions_count

            features['accumulated_game_miss'] = accumulated_game_miss

            features['mean_game_round'] = mean_game_round

            features['mean_game_duration'] = mean_game_duration

            features['mean_game_level'] = mean_game_level

            features['Assessment_mean_event_count'] = Assessment_mean_event_count

            features['Game_mean_event_count'] = Game_mean_event_count

            features['Activity_mean_event_count'] = Activity_mean_event_count

            features['chest_assessment_uncorrect_sum'] = chest_assessment_uncorrect_sum

            

            variety_features = [('var_event_code', event_code_count),

                              ('var_event_id', event_id_count),

                               ('var_title', title_count),

                               ('var_title_event_code', title_event_code_count),

                                ('var_type_world', type_world_count)]

            

            for name, dict_counts in variety_features:

                arr = np.array(list(dict_counts.values()))

                features[name] = np.count_nonzero(arr)

                 

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            

            features['Cauldron_Filler_4025'] = Cauldron_Filler_4025/counter if counter > 0 else 0

            ####################

            Assess_4025 = session[(session.event_code == 4025) & (session.title=='Cauldron Filler (Assessment)')]   

            true_attempts_ = Assess_4025['event_data'].str.contains('true').sum()

            false_attempts_ = Assess_4025['event_data'].str.contains('false').sum()



            cau_assess_accuracy_ = true_attempts_/(true_attempts_+false_attempts_) if (true_attempts_+false_attempts_) != 0 else 0

            Cauldron_Filler_4025 += cau_assess_accuracy_

            

            chest_assessment_uncorrect_sum += len(session[session.event_id=="df4fe8b6"])

            

            Assessment_mean_event_count = (Assessment_mean_event_count + session['event_count'].iloc[-1])/2.0

            ##############################

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

        

        # end session_type=="Assessment"

        

        sessions_count += 1

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title':

                        x = activities_labels[k]

                    counter[x] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')

        type_world_count = update_counters(type_world_count, 'type_world')

        

        assess_4020_acc_dict = get_4020_acc(session , assess_4020_acc_dict)

        game_time_dict[session_type+'_gametime'] = (game_time_dict[session_type+'_gametime'] + (session['game_time'].iloc[-1]/1000.0))/2.0

        

        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
#train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code, activities_map,all_type_world = encode_title(train, test, train_labels)