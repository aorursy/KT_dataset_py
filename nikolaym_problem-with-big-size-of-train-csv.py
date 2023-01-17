import pandas as pd
# train_file_locate - variable with path to train.csv For example 'data/train.csv'
def split_to_separate_earthquake_files(train_file_locate):
    print('Begin')
    chunksize = 10 ** 7
    i = 1
    n = 0
    time_to_falture_pr = chunksize
    dt = pd.DataFrame
    frames = []
    for chunk in pd.read_csv(train_file_locate, chunksize=chunksize):
        n += 1
        print('read chank_'+str(n))
        time_to_falture_begin = chunk['time_to_failure'][(n-1)*chunksize]
        time_to_falture_end = chunk['time_to_failure'][(n-1)*chunksize + chunk.shape[0] - 1]
        if (time_to_falture_begin > time_to_falture_end) and (time_to_falture_begin < time_to_falture_pr) :
            frames.append(chunk)
        else:
            if time_to_falture_begin > time_to_falture_pr:
                print('saving earthquake_'+str(i))   
                fr_to_csv = pd.concat(frames)
                fr_to_csv.to_csv('data/earthquake_'+str(i)+'.csv')
                print('saved earthquake_'+str(i))            
                dt = pd.DataFrame
                frames = []
                i += 1
            else:
                if time_to_falture_begin > time_to_falture_end:
                    frames.append(chunk)
                else:
                    frames.append(chunk.loc[chunk['time_to_failure'] < time_to_falture_end])
                    print('saving_2 earthquake_'+str(i)) 
                    fr_to_csv = pd.concat(frames)
                    fr_to_csv.to_csv('data/earthquake_'+str(i)+'.csv')
                    print('saved_2 earthquake_'+str(i))            
                    dt = pd.DataFrame
                    frames = []
                    i += 1
                    frames.append(chunk.loc[chunk['time_to_failure'] < time_to_falture_end])
        time_to_falture_pr = time_to_falture_end
    print('Ready')
    # return count of earthquake
    return i


