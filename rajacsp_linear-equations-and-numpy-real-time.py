import numpy as np



items_spent = np.array([

        [3, 2],

        [2, 1]

    ])

    

spent_total = np.array([30, 19])

item_value = np.linalg.solve(items_spent, spent_total)



print(item_value)
# Check the solution is right

print(np.allclose(np.dot(items_spent, item_value), spent_total))