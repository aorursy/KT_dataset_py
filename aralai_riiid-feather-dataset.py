import pandas as pd
%%time



dtypes = {

    "row_id": "int64",

    "timestamp": "int64",

    "user_id": "int32",

    "content_id": "int16",

    "content_type_id": "boolean",

    "task_container_id": "int16",

    "user_answer": "int8",

    "answered_correctly": "int8",

    "prior_question_elapsed_time": "float32", 

    "prior_question_had_explanation": "boolean"

}



files = ['train', 'questions', 'lectures', 'example_test', 'example_sample_submission']



for file in files:

    if file=='train':

        data = pd.read_csv("../input/riiid-test-answer-prediction/{0}.csv".format(file), dtype=dtypes)

    else:

        data = pd.read_csv("../input/riiid-test-answer-prediction/{0}.csv".format(file))

    data.to_feather("{0}.feather".format(file))

    print("File: {0} - size: {1}".format(file,data.shape))
