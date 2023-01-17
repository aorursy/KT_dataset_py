!pip install jovian -q --upgrade
import jovian
jovian.commit()
# Updating the notebook

jovian.commit()
jovian.commit(notebook_id="903a04b17036436b843d70443ef5d7ad")
import numpy as np

from utils import sigmoid



inputs = np.array([1, 2, 3, 4, 5, 6, 7, 8])



outputs = sigmoid(inputs)

print(outputs)



np.savetxt("outputs.csv", outputs, delimiter=",")
!cat utils.py
!cat outputs.csv
jovian.commit(files=['utils.py'], artifacts=['outputs.csv'])