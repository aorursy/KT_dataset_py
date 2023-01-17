! cp -a /kaggle/input/catalyst/catalyst/catalyst/install.sh /tmp/install.sh && chmod 777 /tmp/install.sh && /tmp/install.sh /kaggle/input/catalyst/catalyst/catalyst
! ls /kaggle/input/mnistcatalyst
cat /kaggle/input/mnistcatalyst/__init__.py
cat /kaggle/input/mnistcatalyst/experiment.py
cat /kaggle/input/mnistcatalyst/model.py
cat /kaggle/input/mnistcatalyst/dataset.py
cat /kaggle/input/mnistcatalyst/train.yml
cat /kaggle/input/mnistcatalyst/infer.yml
!head /kaggle/input/mnistcatalyst/fold.csv
! catalyst-dl run --config /kaggle/input/mnistcatalyst/train.yml --expdir /kaggle/input/mnistcatalyst/
! catalyst-dl run --config /kaggle/input/mnistcatalyst/infer.yml --expdir /kaggle/input/mnistcatalyst/
! ls /tmp/log
import numpy as np

import pandas as pd



prob = np.load('/tmp/log/infer.logits.npy')

argmax = prob.argmax(axis=1)

pd.DataFrame({

    'ImageId': np.arange(1, len(argmax) + 1),

    'Label': argmax

}).to_csv('submission.csv', index=False)