import numpy as np
import pandas as pd
import plotnine as pn #To avoid universal import
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.Generation = data.Generation.astype("object")
data.head()
dataHold = data.groupby(["Generation"]).mean().reset_index()
(pn.ggplot(dataHold)
 + pn.aes(x="Sp. Atk", y="Sp. Def", color="Generation", label="Generation")
 + pn.geom_path(size=1)
 + pn.geom_point(color="black")
 + pn.geom_text(color="black",
                size=10,
                nudge_y=0.4
               )
 + pn.scale_x_continuous(limits=(65, 78),
                         breaks=np.arange(65, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(67.5, 78),
                         breaks=np.arange(67.5, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      )
           )
 + pn.labs(x="Base Sp. Attack",
           y="Base Sp. Defense",
           color="Gen."
          )
 + pn.ggtitle("Relation between Sp. Attack and Sp. Defense over Time")
)
dataHold = data.groupby(["Generation"]).mean().reset_index()
dataHold = data.groupby(["Generation"]).mean().reset_index()
(pn.ggplot(dataHold)
 + pn.aes(x="Sp. Atk", y="Sp. Def", color="Generation", label="Generation")
 + pn.geom_path(size=1)
 + pn.geom_point(color="black")
 + pn.geom_text(data=dataHold.iloc[[0, 1, 3, 5], :],
                color="black",
                size=10,
                nudge_y=0.4
               )
  + pn.geom_text(data=dataHold.iloc[[2, 4], :],
                color="black",
                size=10,
                nudge_y=-0.4
               )
 + pn.scale_x_continuous(limits=(65, 78),
                         breaks=np.arange(65, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(67.5, 78),
                         breaks=np.arange(67.5, 78, 2.5),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      )
           )
 + pn.labs(x="Base Sp. Attack",
           y="Base Sp. Defense",
           color="Gen."
          )
 + pn.ggtitle("Relation between Sp. Attack and Sp. Defense over Time")
)
"""
Data copied from the ggplot2 documentation (https://ggplot2.tidyverse.org/reference/geom_spoke.html)
The pokemon data isn't ideal for this, so lets create a dataframe!
"""
dataHold = pd.DataFrame({"angle": np.random.uniform(0, 2 * np.pi, 121),
                         "speed": np.random.uniform(0, 0.5, size=121)},
                        index=pd.MultiIndex.from_product([range(11), range(11)], names=["X", "Y"])
                       ).reset_index()
dataHold.head()
(pn.ggplot(dataHold)
 + pn.aes(x="X", y="Y", angle="angle", radius="speed")
 + pn.geom_point(size=0.5,
                 color="white"
                )
 + pn.geom_spoke(pn.aes(color="speed"), 
                 size=1
                )
 + pn.scale_x_continuous(limits=(0, 10),
                         breaks=np.arange(0, 11, 2),
                         expand=(0, 0.5)
                        )
 + pn.scale_y_continuous(limits=(0, 10),
                         breaks=np.arange(0, 11, 2),
                         expand=(0, 0.5)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(size=0.25,
                                       color="white"
                                      ),
            figure_size=(10, 10)
           ) 
 +pn.labs(color="Speed")
 + pn.scale_color_gradient2(low="yellow", mid="orange", high="red", midpoint=0.25)
 + pn.ggtitle("An Arbitrary Vector Space")
)
(pn.ggplot(data)
 + pn.aes(x="Attack", color="Legendary")
 + pn.geom_density(size=1, 
                   trim=False
                  )
 + pn.facet_wrap("~ Generation")
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=np.arange(0, 201, 50),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 0.02),
                         breaks=np.arange(0, 0.021, 0.005),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      ),
            figure_size=(12, 6)
           )  
 + pn.labs(y="Density")
 + pn.ggtitle("Base Attack Stat for Legendary and Non-Legendary\n Pokemon Across Generations\n")
)
(pn.ggplot(data)
 + pn.aes(x="Attack")
 + pn.geom_density(size=1, 
                   trim=False,
                   color="darkslateblue"
                  )
 + pn.facet_grid("Generation ~ Legendary")
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=np.arange(0, 201, 50),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 0.02),
                         breaks=np.arange(0, 0.021, 0.005),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(size=0.25,
                                       color="black"
                                      ),
            figure_size=(6, 12)
           )  
 + pn.labs(y="Density")
 + pn.ggtitle("Base Attack Stat for Legendary and Non-Legendary\n Pokemon Across Generations\n")
)
(pn.ggplot(data)
 + pn.aes(x="Defense", y="Sp. Def")
 + pn.geom_jitter(pn.aes(color="Type 1"),
                  size=1
                 )
 + pn.geom_smooth(size=0.5,
                  color="white",
                  method="lm",
                  fullrange=False,
                  se=False
                 )
 + pn.facet_grid("Generation ~ Legendary")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(size=0.25,
                                       color="white"
                                      ),
            figure_size=(8, 12)
           )  
 + pn.ggtitle("Correlation bettween Defense and Sp. Defense\n by Generation and LEgendary Status\n")
)
