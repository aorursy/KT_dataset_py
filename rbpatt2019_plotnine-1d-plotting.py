import pandas as pd
import plotnine as pn #To avoid universal import
data = pd.read_csv("../input/Pokemon.csv", index_col=0)
data.Generation = data.Generation.astype("object")
data.sample(10)
(pn.ggplot(data)
 + pn.aes(x="Type 1")
 + pn.geom_bar(fill="darkslateblue")
 + pn.xlab("Primary Type")
 + pn.ylab("Number of Pokemon")
 + pn.coord_flip()
 + pn.ggtitle("A Count of Pokemon Type")
 + pn.theme(figure_size=(12, 6))
)
(pn.ggplot(data.groupby(["Type 1"]).Attack.mean().round(2).reset_index())
 + pn.aes(x="Type 1", y="Attack", label="Attack")
 + pn.geom_col(fill="gold")
 + pn.geom_text(ha="center",
                nudge_y=3,
                size=8
               )
 + pn.scale_y_continuous(limits=(0, 130),
                         breaks=(range(0, 131, 10)),
                         expand=(0,0)
                        )
 + pn.theme(axis_text_x=pn.element_text(rotation=45),
            figure_size=(12, 6))
 + pn.ggtitle("Average Base Attack by Primary Type")
 + pn.xlab("Primary Type")
)
(pn.ggplot(data)
 + pn.aes(x="Attack", fill="Generation")
 + pn.geom_histogram(binwidth=5,
                     position=pn.position_stack(reverse=True)
                    )
 + pn.xlab("Base Attack")
 + pn.ylab("Number of Pokemon")
 + pn.scale_y_continuous(limits=(0, 71),
                         breaks=(range(0, 71, 10)),
                         expand=(0,0)
                        )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=(range(0, 201, 10)),
                         expand=(0,0)
                        )
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(figure_size=(12, 6),
            panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      )
           )
)
(pn.ggplot(data)
 + pn.aes(x="Attack", color="Generation")
 + pn.geom_freqpoly(binwidth=5,
                    size=1
                   )
 + pn.xlab("Base Attack")
 + pn.ylab("Number of Pokemon")
 + pn.scale_y_continuous(limits=(0, 21),
                         breaks=(range(0, 21, 5)),
                         expand=(0,0)
                        )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=(range(0, 201, 10)),
                         expand=(0,0)
                        )
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
)
(pn.ggplot(data)
 + pn.aes(x="Attack", color="Generation")
 + pn.geom_density(adjust=0.5, 
                   size=1
                  )
 + pn.scale_x_continuous(limits=(0, 201),
                         breaks=range(0, 201, 10),
                         expand=(0, 0)
                        )
 + pn.xlab("Base Attack")
 + pn.ylab("Number of Pokemon")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black", 
                                       size=0.25
                                      ),
            figure_size=(12, 6)
           )
 )
(pn.ggplot(data)
 + pn.aes(x="Generation", y="Attack")
 + pn.geom_boxplot(notch=True, 
                   varwidth=True,
                   fill="slateblue",
                   size=1
                  )
 + pn.scale_y_continuous(limits=(0, 201),
                         breaks=range(0, 201, 25),
                        )
 + pn.xlab("Generation")
 + pn.ylab("Base Attack")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
)
(pn.ggplot(data)
 + pn.aes(x="Generation", y="Attack")
 + pn.geom_violin(adjust=0.5, 
                  scale="count",
                  draw_quantiles=[0.25, 0.5, 0.75],
                  size=1,
                  fill="Gold"
                 )
 + pn.geom_jitter(color="black",
                 size=1,
                  width=0.2
                 )
 + pn.scale_y_continuous(limits=(0, 201),
                         breaks=range(0, 201, 25),
                        )
 + pn.xlab("Generation")
 + pn.ylab("Base Attack")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
 + pn.coord_flip()
)
(pn.ggplot(data)
 + pn.aes(x="Generation", y="Attack", fill="Legendary")
 + pn.geom_violin(adjust=0.5, 
                  scale="count",
                  draw_quantiles=[0.25, 0.5, 0.75],
                  size=1,
                 )
 + pn.geom_jitter(color="black",
                 size=1,
                  width=0.2
                 )
 + pn.scale_y_continuous(limits=(0, 201),
                         breaks=range(0, 201, 25),
                        )
 + pn.xlab("Generation")
 + pn.ylab("Base Attack")
 + pn.ggtitle("Distribution of Attack by Generation")
 + pn.theme(panel_background=pn.element_rect(fill="white"),
            panel_grid=pn.element_line(color="black",
                                       size=0.25
                                      ), 
            figure_size=(12, 6)
           )
)