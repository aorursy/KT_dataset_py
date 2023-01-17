import numpy as np
import tqdm
def gen_people():
    dates = []
    leap_years = np.arange(1972, 2001, 4)

    for p in range(70):
        y = np.random.randint(1970, 2001)
        m = np.random.randint(1, 13)
        if m in [1,3,5,7,8,10,12]:
            d = np.random.randint(1, 32)
        elif m in [4,6,9,11]:
            d = np.random.randint(1, 31)
        else:
            if y in leap_years:
                d = np.random.randint(1, 30)
            else:
                d = np.random.randint(1, 29)

        dates.append([str(d)+'-'+str(m)])
    
    return np.vstack(dates)
def check_triple(dates):
    d, cnt = np.unique(dates, return_counts=True)
    
    return max(cnt) > 2
bingo = 0
iters = 100000
for i in tqdm.tqdm_notebook(range(iters)):
    if check_triple(gen_people()):
        bingo += 1
probs = bingo / iters
print(probs)