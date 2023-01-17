from xml.etree import ElementTree
tree = ElementTree.parse('../input/mesa-sleep-0001-nsrr.xml')
root = tree.getroot()

import pandas as pd
data = pd.DataFrame([])
for att in root.find('ScoredEvents'):
    try:
        #         print(att.find('ScoredEvent').find('Start').text)
        #     first = att.find('ScoredEvent').text
        #         print("==")
        #         print(att.find('Start').text)
        #         print(att.find('Duration').text)
        data = data.append([[att.find('Start').text, att.find('Duration').text]])
        #         for subatt in att.find('ScoredEvent'):
        #             print("\\", subatt)
        #             sec = subatt.find('Start').text
        #             print(sec)
    except Exception as ex:
        pass
    
    
data.columns = ['Start Time', 'Duration']
data
import pyedflib
import numpy as np

f = pyedflib.EdfReader("../input/mesa-sleep-0001.edf")
n = f.signals_in_file
signal_labels = f.getSignalLabels()
sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i, :] = f.readSignal(i)
import pip

