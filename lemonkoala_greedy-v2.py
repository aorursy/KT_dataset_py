import numpy   as np

import pandas  as pd
brats = pd.read_csv("../input/child_wishlist_v2.csv", header=None, index_col=0).as_matrix()

gifts = pd.read_csv("../input/gift_goodkids_v2.csv",  header=None, index_col=0).as_matrix()
TRIPS_COUNT =  5_001

TWINS_COUNT = 40_000

TWINS_START = TRIPS_COUNT

TWINS_END   = TWINS_START + TWINS_COUNT

GIFTS_LIMIT =  1_000



BRAT_PREF_COUNT = brats.shape[1]

GIFT_PREF_COUNT = gifts.shape[1]
assigned = [None] * len(brats)

gift_cnt = [0]    * len(gifts)



def assign(brat_id, gift_id):

    assigned[brat_id]  = gift_id

    gift_cnt[gift_id] += 1

    

def find_best_preferred(brat_id, copies=0):

    for preference_order in range(BRAT_PREF_COUNT):

        preference = brats[brat_id, preference_order]

        if gift_cnt[preference] + copies < GIFTS_LIMIT:

            return preference

    return np.argmin(gift_cnt) # least_popular_gift



def display_progress(curr_id, total=len(brats)):

    completed = curr_id + 1

    percent   = completed / total * 100

    if percent % 10 == 0:

        percent_str = str(percent).rjust(5)

        print(f"Completed: {percent_str}%")
for trip_1 in range(0, TRIPS_COUNT, 3):

    trip_1_preference = find_best_preferred(trip_1, 2)

    assign(trip_1,     trip_1_preference)

    assign(trip_1 + 1, trip_1_preference)

    assign(trip_1 + 2, trip_1_preference)
for twin_1 in range(TWINS_START, TWINS_END, 2):

    twin_1_preference = find_best_preferred(twin_1, 1)

    assign(twin_1,     twin_1_preference)

    assign(twin_1 + 1, twin_1_preference)
for brat_id in range(TWINS_END, len(brats)):

    best_gift = find_best_preferred(brat_id)

    assign(brat_id, best_gift)

    

    display_progress(brat_id)
print("Quantity of Gifts:")

pd.Series(gift_cnt).value_counts()
def calc_brat_happiness(brat_id, gift_id):

    return calc_happiness(brats, brat_id, gift_id)



def calc_gift_happiness(gift_id, brat_id):

    return calc_happiness(gifts, gift_id, brat_id)



def calc_happiness(matrix, a, b):

    if b in matrix[a]:

        rank = len(matrix[a]) - list(matrix[a]).index(b)

        return rank * 2

    else:

        return -1



BRAT_MAX_HAPPINESS = 2 * BRAT_PREF_COUNT

GIFT_MAX_HAPPINESS = 2 * GIFT_PREF_COUNT
rows = []



for brat_id, gift_id in enumerate(assigned):

    row = {

        "BratId": brat_id,

        "GiftId": gift_id,

        "NCH":    calc_brat_happiness(brat_id, gift_id) / BRAT_MAX_HAPPINESS,

        "NSH":    calc_gift_happiness(gift_id, brat_id) / GIFT_MAX_HAPPINESS

    }

    rows.append(row)



    display_progress(brat_id)
scores = pd.DataFrame(rows)

scores.head()
ANCH = scores.NCH.mean()

ANSH = scores.NSH.mean()

ANTH = ANCH ** 3 + ANSH ** 3



print("Average child happiness: %.9f" % ANCH)

print("Average santa happiness: %.9f" % ANSH)

print("Average total happiness: %.9f" % ANTH)
submission = pd.DataFrame({

    "ChildId": range(len(brats)),

    "GiftId":  assigned

})
submission.to_csv("greedy_v2.csv", index=False)