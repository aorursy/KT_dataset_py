def map_at_k(y_true, y_pred, k=2):

    """

    y_true: list of ground truths

    y_pred: list of y_predictions

    k: value to set depending of requirements. Default at 2

    """

    ap = []

    for g,p in zip(y_true,y_pred):

        assert len(p)>=k, f"Length of each prediction must be equal or greater than {k}!"

        for i in range(k):

            if g == p[i]:

                ap.append(1/(i+1))

                break

        else:

            ap.append(0)

        

    return(sum(ap)/len(ap))
# all correct answers in the first try

ground_truth = [1,3,0,3,2]

predictions = [(1,2),(3,1),(0,3),(3,1),(2,0)]

map_at_k(ground_truth, predictions)
# more correct answers in the first try than second try

ground_truth = [1,3,0,3,2]

predictions = [(1,3),(3,1),(0,1),(1,3),(0,2)]

map_at_k(ground_truth, predictions)
# more correct answers in the second try than first try

ground_truth = [1,3,0,3,2]

predictions = [(1,3),(3,1),(1,0),(1,3),(0,2)]

map_at_k(ground_truth, predictions)
# all correct answers in the second try

ground_truth = [1,3,0,3,2]

predictions = [(3,1),(1,3),(2,0),(1,3),(0,2)]

map_at_k(ground_truth, predictions)
# with mostly inaccurate answers

ground_truth = [1,3,0,3,1]

predictions = [(1,2),(0,3),(3,1),(2,0),(3,2)]

map_at_k(ground_truth, predictions)
# no right answers

ground_truth = [1,3,0,3,2]

predictions = [(3,2),(2,1),(3,1),(2,1),(3,0)]

map_at_k(ground_truth, predictions)