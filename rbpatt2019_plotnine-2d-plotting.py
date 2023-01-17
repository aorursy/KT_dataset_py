import pandas as pd
import plotnine as pn #To avoid universal import
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.Generation = data.Generation.astype("object")
data.head()
(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense",)
 + pn.geom_point(size=0.9,
                  color="darkslateblue"
                 )
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      )
           )
)
(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense", color="Generation")
 + pn.geom_jitter(size=0.9)
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8)
           )
)
(pn.ggplot(data[data.Generation == 1]) #Downsampling
 + pn.aes(x="Attack", y="Defense", color="Type 1", label="Name")
 + pn.geom_text(size=8,
                #check_overlap=True
               )
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8)
           )
)
(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense", color="Generation")
 + pn.geom_jitter(size=0.9)
 + pn.geom_smooth(fullrange=False,
                  se=False, #hides confidence interval
                  method="lm" #to force linear fitting
                 )
 + pn.geom_rug(sides="tr")
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
  + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 50),
                         expand=(0, 0)
                        )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8)
           )
)
(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense")
 + pn.geom_bin2d(binwidth=10,
                 drop=False #to fill in all grey background
                )
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.scale_x_continuous(limits=(0, 251),
                         breaks=range(0, 251, 20),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 20),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
             panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ),
             figure_size=(8,8)
            )
 + pn.coord_fixed(ratio=1) #ensures square boxes
)
(pn.ggplot(data.dropna()) #removes pokemon of only 1 type
 + pn.aes(x="Type 1", y="Type 2")
 + pn.geom_bin2d()
 + pn.ggtitle("Attack and Defense by Generation")
 + pn.theme(figure_size=(8,8),
            
           )
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8),
            axis_text=pn.element_text(rotation=45)
           )
 + pn.coord_fixed(ratio=1) #ensures square boxes
 + pn.scale_fill_cmap("RdPu")
 + pn.xlab("Primary Type")
 + pn.ylab("Secondary Type")
)
(pn.ggplot(data)
 + pn.aes(x="Attack", y="Defense")
 + pn.geom_jitter(size=1)
 + pn.geom_density_2d(color="darkslateblue",
                      size=1
                     )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=range(0, 201, 20),
                         expand=(0, 0)
                        )
 + pn.scale_y_continuous(limits=(0, 251),
                         breaks=range(0, 251, 20),
                         expand=(0, 0)
                        )
 + pn.theme(panel_background=pn.element_rect(fill="white"),
             panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ),
             figure_size=(8,8)
            )
 + pn.ggtitle("Attack and Defense Density Estimates")
)

(pn.ggplot(pd.melt(data.groupby(["Generation"]).mean().reset_index(),
                   id_vars=["Generation"], value_vars=["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
                  )
          )
 + pn.aes(x="Generation", y="value", color="variable")
 + pn.geom_line()
 + pn.scale_x_discrete(limits=("1", "2", "3", "4", "5", "6"),
                       expand=(0,0))
 + pn.scale_y_continuous(limits=(60, 86),
                         breaks=range(60, 86, 5),
                         expand=(0, 0)
                        )
 + pn.labs(y="Base Stat Value",
           color="Stat"
          )
 + pn.ggtitle("Average Base Stats Across Generations")
 + pn.theme(panel_background=pn.element_rect(fill="black"),
            panel_grid=pn.element_line(color="white",
                                       size=0.25
                                      ),
            figure_size=(8,8),
           )
)
pd.melt(data.groupby(["Generation"]).mean().reset_index(),
        id_vars=["Generation"], value_vars=["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
       )
