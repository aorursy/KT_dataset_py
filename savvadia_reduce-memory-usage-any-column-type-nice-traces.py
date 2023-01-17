import numpy as np
import pandas as pd
def reduce_mem_usage(df, comment=None, traces=True):
    """ Iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage. 
        Can be called multiple times for the same dataframe.
        Use 'comment' to keep a log about changes.
        Set 'traces' to False to minimize tracing
        
        NaN, INF and NINF values are not converted, so for better results
        get rid of these values/rows if acceptable. E.g. with clean_nan(df)
    """
    start_mem = df.memory_usage().sum() 
    if traces:
        print('reduce_mem_usage for {:.2f} MB dataframe'.format(start_mem/1024/1024))    

    changes = {}
    skipped = []
    
    for col in df.columns:
        try:
            col_type = df[col].dtype
            
            if col_type is object:
                skipped.append("%s/%s" % (col, col_type))
                continue

            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            if col_type is not df[col].dtype:
                changed_types = "{:10s} --> {:s}".format(str(col_type), str(df[col].dtype))
                if changed_types not in changes:
                    changes[changed_types] = [col]
                else:
                    changes[changed_types].append(col)
            # keep as is otherwise; we can only destroy things if we change type
        except:
            skipped.append("%s/%s" % (col, col_type))
            continue

    if traces:
        print("skipped: ", ", ".join(skipped).replace("<class '",'').replace("'>",''))

    end_mem = df.memory_usage().sum() 
    print('Memory usage {:.2f} MB ==> {:.2f} MB (reduced by {:.1f}%)'.format(start_mem/1024/1024, end_mem/1024/1024, 100 * (start_mem - end_mem) / start_mem), end="")
    if comment:
        print(' comment:', comment)
    else:
        print()

    if traces:
        for t, cols in changes.items():
            print("\t{:25s} {:s}".format(t, ", ".join(cols)))
            
    return df

def clean_nan(df):
    print("Setting NaN, Inf, NINF to 0 in dataframe")
    for col in df.columns:
        try:
            df.loc[df[col] == np.Inf, col] = 0
            df.loc[df[col] == np.NINF, col] = 0
            df.loc[np.isnan(df[col]), col] = 0
        except:
            print("skip cleaning nan for", col)

    return df
# Now let's try it! 

# Generate some data to play with
df = pd.DataFrame(np.random.randint(0,100,size=(10000, 4)), columns=list('ABCD'))
df['i16'] = df['A'] * 100
df['i16_more'] = df['A'] * 200
df['i32'] = df['A'] * 1000
df['f16'] = df['A'] / 1.33
df['f32'] = df['i32'] / 1.33
df['cat'] = df['A'].astype('category')
df['obj'] = df['A'].astype(object)
df['str'] = 'A' + df['A'].apply(str)
df['nan'] = df['A']; df.loc[df['nan'] > 50, 'nan'] = np.NaN
df['inf'] = df['A']; df.loc[df['inf'] > 50, 'inf'] = np.Inf
df['nin'] = df['A']; df.loc[df['nin'] > 50, 'nin'] = np.NINF

df.describe()
df.head()
# before the memory optimization
df.dtypes
df = reduce_mem_usage(df, "initial")
df.head()
df = reduce_mem_usage(df, "safely call one more time")
df = reduce_mem_usage(df, "silent", False)
# default behavior
df = reduce_mem_usage(df)
# column types after the optimization
df.dtypes
# memory usage grows quickly when adding own features
if 'group' not in df:
    df['group'] = df['A'] % 5
    print("get group size feature")
    agg = df.groupby(['group']).size().reset_index(name='group_size')
    df = df.merge(agg, how='left', on=['group'])

df.head()
df = reduce_mem_usage(df, "after features were added")
df.dtypes
df = clean_nan(df)
df = reduce_mem_usage(df, "after NaN's, INF's and NINF's were removed")
df.dtypes
df.head()