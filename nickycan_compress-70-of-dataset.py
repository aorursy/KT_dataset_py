import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
INT8_MIN    = np.iinfo(np.int8).min

INT8_MAX    = np.iinfo(np.int8).max

INT16_MIN   = np.iinfo(np.int16).min

INT16_MAX   = np.iinfo(np.int16).max

INT32_MIN   = np.iinfo(np.int32).min

INT32_MAX   = np.iinfo(np.int32).max



FLOAT16_MIN = np.finfo(np.float16).min

FLOAT16_MAX = np.finfo(np.float16).max

FLOAT32_MIN = np.finfo(np.float32).min

FLOAT32_MAX = np.finfo(np.float32).max



def memory_usage(data, detail=1):

    if detail:

        display(data.memory_usage())

    memory = data.memory_usage().sum() / (1024*1024)

    print("Memory usage : {0:.2f}MB".format(memory))

    return memory



def compress_dataset(data):

    """

        Compress datatype as small as it can

        Parameters

        ----------

        path: pandas Dataframe



        Returns

        -------

            None

    """

    memory_before_compress = memory_usage(data, 0)

    print()

    length_interval      = 50

    length_float_decimal = 4



    print('='*length_interval)

    for col in data.columns:

        col_dtype = data[col][:100].dtype



        if col_dtype != 'object':

            print("Name: {0:24s} Type: {1}".format(col, col_dtype))

            col_series = data[col]

            col_min = col_series.min()

            col_max = col_series.max()



            if col_dtype == 'float64':

                print(" variable min: {0:15s} max: {1:15s}".format(str(np.round(col_min, length_float_decimal)), str(np.round(col_max, length_float_decimal))))

                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):

                    data[col] = data[col].astype(np.float16)

                    print("  float16 min: {0:15s} max: {1:15s}".format(str(FLOAT16_MIN), str(FLOAT16_MAX)))

                    print("compress float64 --> float16")

                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):

                    data[col] = data[col].astype(np.float32)

                    print("  float32 min: {0:15s} max: {1:15s}".format(str(FLOAT32_MIN), str(FLOAT32_MAX)))

                    print("compress float64 --> float32")

                else:

                    pass

                memory_after_compress = memory_usage(data, 0)

                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))

                print('='*length_interval)



            if col_dtype == 'int64':

                print(" variable min: {0:15s} max: {1:15s}".format(str(col_min), str(col_max)))

                type_flag = 64

                if (col_min > INT8_MIN/2) and (col_max < INT8_MAX/2):

                    type_flag = 8

                    data[col] = data[col].astype(np.int8)

                    print("     int8 min: {0:15s} max: {1:15s}".format(str(INT8_MIN), str(INT8_MAX)))

                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):

                    type_flag = 16

                    data[col] = data[col].astype(np.int16)

                    print("    int16 min: {0:15s} max: {1:15s}".format(str(INT16_MIN), str(INT16_MAX)))

                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):

                    type_flag = 32

                    data[col] = data[col].astype(np.int32)

                    print("    int32 min: {0:15s} max: {1:15s}".format(str(INT32_MIN), str(INT32_MAX)))

                    type_flag = 1

                else:

                    pass

                memory_after_compress = memory_usage(data, 0)

                print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))

                if type_flag == 32:

                    print("compress (int64) ==> (int32)")

                elif type_flag == 16:

                    print("compress (int64) ==> (int16)")

                else:

                    print("compress (int64) ==> (int8)")

                print('='*length_interval)



    print()

    memory_after_compress = memory_usage(data, 0)

    print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
data = train.append(test)
compress_dataset(data)