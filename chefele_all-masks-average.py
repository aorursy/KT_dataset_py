import numpy as np
import matplotlib.pyplot as plt
import glob
def plot_patient_avg_mask(patient):
    patient_path = "../input/train/%i_*mask.tif" % patient
    patient_masks = np.array([plt.imread(f) for f in glob.glob(patient_path)])
    plt.title('Average Mask for Patient: %i' % patient)
    plt.imshow(patient_masks.sum(axis=0))
    plt.show()
for patient in range(1,47+1):
    plot_patient_avg_mask(patient)
