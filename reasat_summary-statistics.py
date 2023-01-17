# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Importing necessary libraries
import os
import glob
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
data_dir=os.path.join('..','input')
print(os.listdir(data_dir))

# Any results you write to the current directory are saved as output.
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))

paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
# path_label_train_a=os.path.join(data_dir,'training-a.csv')
# path_label_train_b=os.path.join(data_dir,'training-b.csv')
# path_label_train_e=os.path.join(data_dir,'training-e.csv')
# path_label_train_c=os.path.join(data_dir,'training-c.csv')
# path_label_train_d=os.path.join(data_dir,'training-d.csv')
total_test=len(paths_test_a)+len(paths_test_b)+len(paths_test_c)+ len(paths_test_d)+\
len(paths_test_e)+ len(paths_test_f)+len(paths_test_auga)+ len(paths_test_augc)
df_summary=pd.DataFrame(data={
                        'Train Samples':[len(paths_train_a),len(paths_train_b),
                                         len(paths_train_c),len(paths_train_d),
                                         len(paths_train_e),0,0,0,
                                        ],
                        'Test Samples':[len(paths_test_a),len(paths_test_b),
                                        len(paths_test_c), len(paths_test_d),
                                        len(paths_test_e), len(paths_test_f),
                                        len(paths_test_auga), len(paths_test_augc),
                                       ],
                        'Total':[len(paths_train_a)+len(paths_test_a),
                                 len(paths_train_b)+len(paths_test_b),
                                 len(paths_train_c)+len(paths_test_c),
                                 len(paths_train_d)+len(paths_test_d),
                                 len(paths_train_e)+len(paths_test_e),
                                 len(paths_test_f),len(paths_test_auga), len(paths_test_augc)
                                ],

                        'Test Ratio':[len(paths_test_a)/total_test,len(paths_test_b)/total_test,
                                      len(paths_test_c)/total_test, len(paths_test_d)/total_test,
                                      len(paths_test_e)/total_test, len(paths_test_f)/total_test,
                                      len(paths_test_auga)/total_test,len(paths_test_augc)/total_test
                                    ],
                        },
    
                        index=['a','b','c','d','e','f','auga','augc'],
                        columns=['Train Samples',
                                 'Test Samples',
                                 'Total',
                                 'Test Ratio'
                                ]                    
            )
df_summary
