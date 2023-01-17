dist = Counter(reduce_train['accuracy_group'])

for k in dist:

    dist[k] /= len(reduce_train)

reduce_train['accuracy_group'].hist()



acum = 0

bound = {}

for i in range(3):

    acum += dist[i]

    bound[i] = np.percentile(final_pred, acum * 100)

print(bound)



def classify(x):

    if x <= bound[0]:

        return 0

    elif x <= bound[1]:

        return 1

    elif x <= bound[2]:

        return 2

    else:

        return 3

    

final_pred = np.array(list(map(classify, final_pred)))



sample_submission['accuracy_group'] = final_pred.astype(int)

sample_submission.to_csv('submission.csv', index=False)

sample_submission['accuracy_group'].value_counts(normalize=True)