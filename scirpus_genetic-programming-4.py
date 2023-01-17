import numpy as np

import pandas as pd

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('../input/Iris.csv',index_col=0)
def GP(data):

    p = (2.000000 +

        1.000000 * (np.floor((0.968085 * np.floor(((data[

            "PetalLengthCm"] + np.round((((-(

            9.869604)) + np.minimum(

            (np.minimum((((data[

                        "PetalLengthCm"

                    ] +

                    (

                        data[

                            "SepalWidthCm"

                        ] *

                        np

                        .cos(

                            (

                                (

                                    data[

                                        "PetalLengthCm"

                                    ] +

                                    data[

                                        "PetalWidthCm"

                                    ]

                                ) /

                                2.0

                            )

                        )

                    )

                ) /

                2.0)), (

                np.cos(((

                        data[

                            "PetalWidthCm"

                        ] +

                        9.869604

                    ) /

                    2.0

                ))))), (9.869604)

        )) / 2.0))) / 2.0))))))

    return p
data.head()
le = LabelEncoder()

data.Species = le.fit_transform(data.Species)
print("Accuracy: %.2f" % accuracy_score(data.Species,GP(data)))