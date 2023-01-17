import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.listdir('../input'))
df_fixed = pd.read_csv('../input/accuracy_fixed_adam.csv')
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(range(len(df_fixed['train_accuracy'])), df_fixed['train_accuracy'], label='Training Accuracy')
ax.plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Test Accuracy')
plt.legend()
plt.show()
print('Test Accuracy: ', '%.2f'%(df_fixed['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_fixed['test_accuracy'].values)*100), '%')
df_clr_2 = pd.read_csv('../input/accuracy_clr_stepsize_2.csv')
df_clr_5 = pd.read_csv('../input/accuracy_clr_stepsize_5.csv')
df_clr_8 = pd.read_csv('../input/accuracy_clr_stepsize_8.csv')

fig, ax = plt.subplots(1, 3, figsize=(20,5), sharey=True)

ax[0].plot(range(len(df_clr_2['train_accuracy'])), df_clr_2['train_accuracy'], label='Training Accuracy')
ax[0].plot(range(len(df_clr_2['test_accuracy'])), df_clr_2['test_accuracy'], label='Test Accuracy')
ax[0].set_title('Stepsize = 2')
ax[0].legend()

ax[1].plot(range(len(df_clr_5['train_accuracy'])), df_clr_5['train_accuracy'], label='Training Accuracy')
ax[1].plot(range(len(df_clr_5['test_accuracy'])), df_clr_5['test_accuracy'], label='Test Accuracy')
ax[1].set_title('Stepsize = 5')
ax[1].legend()

ax[2].plot(range(len(df_clr_8['train_accuracy'])), df_clr_8['train_accuracy'], label='Training Accuracy')
ax[2].plot(range(len(df_clr_8['test_accuracy'])), df_clr_8['test_accuracy'], label='Test Accuracy')
ax[2].set_title('Stepsize = 8')
ax[2].legend()

plt.show()

print('Stepsize = 2')
print('Test Accuracy: ', '%.2f'%(df_clr_2['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_clr_2['test_accuracy'].values)*100), '%')

print('\nStepsize = 5')
print('Test Accuracy: ', '%.2f'%(df_clr_5['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_clr_5['test_accuracy'].values)*100), '%')

print('\nStepsize = 8')
print('Test Accuracy: ', '%.2f'%(df_clr_8['test_accuracy'].values[-1]*100), '%')
print('Maximum Test Accuracy: ', '%.2f'%(max(df_clr_8['test_accuracy'].values)*100), '%')
fig, ax = plt.subplots(1, 3, figsize=(20,5), sharey=True)

ax[0].plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Fixed')
ax[0].plot(range(len(df_clr_2['test_accuracy'])), df_clr_2['test_accuracy'], label='Cyclical')
ax[0].set_title('Stepsize = 2')
ax[0].legend()

ax[1].plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Fixed')
ax[1].plot(range(len(df_clr_5['test_accuracy'])), df_clr_5['test_accuracy'], label='Cyclical')
ax[1].set_title('Stepsize = 5')
ax[1].legend()

ax[2].plot(range(len(df_fixed['test_accuracy'])), df_fixed['test_accuracy'], label='Fixed')
ax[2].plot(range(len(df_clr_8['test_accuracy'])), df_clr_8['test_accuracy'], label='Cyclical')
ax[2].set_title('Stepsize = 8')
ax[2].legend()

plt.show()